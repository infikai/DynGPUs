import asyncio
import aiohttp
import argparse
import time
import random
import re
import pandas as pd
from transformers import AutoTokenizer

# ==============================================================================
# 1. METRICS PARSER (Prometheus Format)
# ==============================================================================
async def get_vllm_metrics(session, metrics_url):
    """
    Fetches vLLM metrics and extracts running/waiting counts.
    Returns: (num_running, num_waiting, gpu_cache_usage)
    """
    try:
        async with session.get(metrics_url) as resp:
            if resp.status != 200:
                return -1, -1, -1
            text = await resp.text()
            
            # Simple regex to find the specific gauges
            # Note: vLLM metric names might vary slightly by version, these are standard.
            running = re.search(r'vllm_num_requests_running\{[^}]*\}\s+(\d+\.?\d*)', text)
            waiting = re.search(r'vllm_num_requests_waiting\{[^}]*\}\s+(\d+\.?\d*)', text)
            gpu_cache = re.search(r'vllm_gpu_cache_usage_perc\{[^}]*\}\s+(\d+\.?\d*)', text)
            
            r_val = float(running.group(1)) if running else 0
            w_val = float(waiting.group(1)) if waiting else 0
            g_val = float(gpu_cache.group(1)) if gpu_cache else 0
            
            return r_val, w_val, g_val
    except Exception:
        return -1, -1, -1

# ==============================================================================
# 2. BACKGROUND LOAD (Dynamic)
# ==============================================================================
class BackgroundLoadManager:
    def __init__(self, session, url, tokenizer, min_input=128, max_input=2048):
        self.session = session
        self.url = url
        self.tokenizer = tokenizer
        self.min_input = min_input
        self.max_input = max_input
        self.active_tasks = set()
        self.target_concurrency = 0
        self.running = False

        # Pre-calc buffer
        print("  [LoadManager] Pre-calculating token buffer...")
        base_text = "The quick brown fox jumps over the lazy dog. "
        base_ids = tokenizer.encode(base_text, add_special_tokens=False)
        self.token_buffer = []
        while len(self.token_buffer) < max_input + 50:
            self.token_buffer.extend(base_ids)

    async def _worker(self):
        while self.running:
            current_load = len(self.active_tasks)
            if current_load < self.target_concurrency:
                deficit = self.target_concurrency - current_load
                # Spawn faster to fill queue
                for _ in range(deficit):
                    task = asyncio.create_task(self._send_req())
                    self.active_tasks.add(task)
                    task.add_done_callback(self.active_tasks.discard)
            await asyncio.sleep(0.01)

    async def _send_req(self):
        try:
            input_len = random.randint(self.min_input, self.max_input)
            slice_ids = self.token_buffer[:input_len]
            prompt_text = self.tokenizer.decode(slice_ids, skip_special_tokens=True)
            
            payload = {
                "prompt": prompt_text,
                "max_tokens": random.randint(10, 100), # Short output to keep turnover high
                "temperature": 0.9,
                "stream": False
            }
            async with self.session.post(self.url, json=payload) as resp:
                await resp.read()
        except:
            pass

    def set_concurrency(self, n):
        self.target_concurrency = n

    def start(self):
        self.running = True
        return asyncio.create_task(self._worker())

    def stop(self):
        self.running = False

# ==============================================================================
# 3. EXPERIMENT LOOP
# ==============================================================================
async def run_experiment(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # We will sweep concurrency to force different Queue states
    # We go higher here to force "Waiting" requests
    concurrency_sweep = [0, 10, 20, 40, 60, 80] 
    
    # Fixed probe length to isolate Queue Time impact
    # (If input length varies, TTFT changes due to prefill, confusing the data)
    PROBE_LEN = 1024 
    probe_prompt = tokenizer.decode(
        tokenizer.encode("test " * 1000, add_special_tokens=False)[:PROBE_LEN], 
        skip_special_tokens=True
    )

    data_points = [] # Stores: (Running, Waiting, GPU_Cache, TTFT)

    async with aiohttp.ClientSession() as session:
        load_manager = BackgroundLoadManager(session, args.url, tokenizer)
        bg_task = load_manager.start()

        for conc in concurrency_sweep:
            print(f"\n--- Setting Concurrency: {conc} ---")
            load_manager.set_concurrency(conc)
            
            # Allow queue to build up
            print("  Warming up / Stabilizing...")
            await asyncio.sleep(args.warmup)

            print(f"  Sending {args.probes} probes...")
            for i in range(args.probes):
                # 1. SNAPSHOT METRICS (Before sending)
                # We want to know the state of the queue *when we joined it*
                n_running, n_waiting, gpu_cache = await get_vllm_metrics(session, args.metrics_url)
                
                # 2. SEND PROBE
                start_t = time.time()
                ttft = None
                try:
                    payload = {
                        "prompt": probe_prompt,
                        "max_tokens": 10,
                        "stream": True,
                        "ignore_eos": True
                    }
                    async with session.post(args.url, json=payload) as resp:
                        if resp.status == 200:
                            async for chunk in resp.content:
                                if ttft is None:
                                    ttft = time.time() - start_t
                                    break # Got TTFT, abort
                except Exception as e:
                    print(f"Probe failed: {e}")

                # 3. LOG DATA
                if ttft is not None:
                    # Conversion to ms
                    ttft_ms = ttft * 1000
                    print(f"    Probe {i}: Waiting={n_waiting:.0f} | Running={n_running:.0f} | TTFT={ttft_ms:.2f}ms")
                    data_points.append({
                        "Running": n_running,
                        "Waiting": n_waiting,
                        "GPU_Cache_Usage": gpu_cache,
                        "TTFT_ms": ttft_ms
                    })
                
                # Small random delay to sample different queue states
                await asyncio.sleep(random.uniform(0.2, 1.0))

        load_manager.stop()
        await bg_task

    # ==========================================
    # 4. SAVE & ANALYZE
    # ==========================================
    df = pd.DataFrame(data_points)
    csv_filename = "queue_time_data.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nData saved to {csv_filename}")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:8002/v1/completions")
    parser.add_argument("--metrics-url", type=str, default="http://localhost:8002/metrics")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--probes", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    
    args = parser.parse_args()
    asyncio.run(run_experiment(args))