import asyncio
import aiohttp
import argparse
import time
import random
import re
import pandas as pd
from transformers import AutoTokenizer

# ==============================================================================
# 1. METRICS PARSER (Updated for vllm: format)
# ==============================================================================
async def get_vllm_metrics(session, metrics_url):
    """
    Fetches vLLM metrics and extracts running/waiting counts.
    Matches format: vllm:num_requests_running{...} 48.0
    """
    try:
        async with session.get(metrics_url) as resp:
            if resp.status != 200:
                print(f"Metrics Error: HTTP {resp.status}")
                return -1, -1, -1
            text = await resp.text()
            
            # --- UPDATED REGEX ---
            # Matches "vllm:num_requests_running" followed by any labels {...} and then the number
            running = re.search(r'vllm:num_requests_running\{[^}]*\}\s+(\d+\.?\d*)', text)
            waiting = re.search(r'vllm:num_requests_waiting\{[^}]*\}\s+(\d+\.?\d*)', text)
            gpu_cache = re.search(r'vllm:kv_cache_usage_perc\{[^}]*\}\s+(\d+\.?\d*)', text)
            
            # Extract values safely
            r_val = float(running.group(1)) if running else 0.0
            w_val = float(waiting.group(1)) if waiting else 0.0
            g_val = float(gpu_cache.group(1)) if gpu_cache else 0.0
            
            return r_val, w_val, g_val
    except Exception as e:
        print(f"Metrics Parse Error: {e}")
        return -1, -1, -1

# ==============================================================================
# 2. BACKGROUND LOAD (Dynamic)
# ==============================================================================
class BackgroundLoadManager:
    def __init__(self, session, url, tokenizer, min_input=500, max_input=2000):
        self.session = session
        self.url = url
        self.tokenizer = tokenizer
        self.min_input = min_input
        self.max_input = max_input
        self.active_tasks = set()
        self.target_concurrency = 0
        self.running = False

        # Pre-calc buffer for fast token slicing
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
                # Spawn tasks to fill deficit
                for _ in range(deficit):
                    task = asyncio.create_task(self._send_req())
                    self.active_tasks.add(task)
                    task.add_done_callback(self.active_tasks.discard)
            await asyncio.sleep(0.01)

    async def _send_req(self):
        try:
            # Dynamic input length for background noise
            input_len = random.randint(self.min_input, self.max_input)
            slice_ids = self.token_buffer[:input_len]
            prompt_text = self.tokenizer.decode(slice_ids, skip_special_tokens=True)
            
            payload = {
                "prompt": prompt_text,
                "max_tokens": random.randint(10, 100), 
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
    
    # Sweep concurrency to force waiting states
    # Adjusted: We likely need high concurrency to see 'waiting' requests if GPU is powerful
    concurrency_sweep = [10, 20, 40, 50, 55, 60, 65, 70, 80, 90]
    
    # Use a fixed probe length to ensure TTFT variations are ONLY due to queueing
    PROBE_LEN = 1024 
    probe_prompt = tokenizer.decode(
        tokenizer.encode("test " * 1000, add_special_tokens=False)[:PROBE_LEN], 
        skip_special_tokens=True
    )

    data_points = [] 

    async with aiohttp.ClientSession() as session:
        load_manager = BackgroundLoadManager(session, args.url, tokenizer)
        bg_task = load_manager.start()

        for conc in concurrency_sweep:
            print(f"\n--- Setting Background Concurrency: {conc} ---")
            load_manager.set_concurrency(conc)
            
            print(f"  > Waiting {args.warmup}s for queue to stabilize...")
            await asyncio.sleep(args.warmup)

            print(f"  > Sending {args.probes} probes...")
            for i in range(args.probes):
                # 1. GET METRICS SNAPSHOT
                n_run, n_wait, kv_usage = await get_vllm_metrics(session, args.metrics_url)
                
                # 2. SEND PROBE & MEASURE TTFT
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
                                    break # Got TTFT, abort stream
                except Exception as e:
                    print(f"Probe failed: {e}")

                # 3. LOG
                if ttft is not None:
                    ttft_ms = ttft * 1000
                    # Print status for real-time monitoring
                    print(f"    [Probe {i}] Wait:{n_wait:3.0f} | Run:{n_run:3.0f} | KV:{kv_usage:.2f} -> TTFT: {ttft_ms:.2f}ms")
                    
                    data_points.append({
                        "Running_Reqs": n_run,
                        "Waiting_Reqs": n_wait,
                        "KV_Cache_Usage": kv_usage,
                        "TTFT_ms": ttft_ms
                    })
                
                # Random sleep to sample different states
                await asyncio.sleep(random.uniform(0.2, 0.8))

        load_manager.stop()
        await bg_task

    # Save to CSV
    df = pd.DataFrame(data_points)
    csv_filename = "queue_v_ttft.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nData saved to {csv_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1/completions")
    parser.add_argument("--metrics-url", type=str, default="http://localhost:8002/metrics")
    parser.add_argument("--model-path", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--probes", type=int, default=15, help="Probes per concurrency level")
    parser.add_argument("--warmup", type=int, default=8, help="Warmup seconds")
    
    args = parser.parse_args()
    asyncio.run(run_experiment(args))