import asyncio
import aiohttp
import argparse
import time
import random
import re
import pandas as pd
from transformers import AutoTokenizer

# ==============================================================================
# 1. METRICS PARSER
# ==============================================================================
async def get_vllm_metrics(session, metrics_url):
    """
    Fetches vLLM metrics and extracts running/waiting counts.
    """
    try:
        async with session.get(metrics_url) as resp:
            if resp.status != 200:
                print(f"Metrics Error: HTTP {resp.status}")
                return -1, -1, -1
            text = await resp.text()
            
            running = re.search(r'vllm:num_requests_running\{[^}]*\}\s+(\d+\.?\d*)', text)
            waiting = re.search(r'vllm:num_requests_waiting\{[^}]*\}\s+(\d+\.?\d*)', text)
            gpu_cache = re.search(r'vllm:kv_cache_usage_perc\{[^}]*\}\s+(\d+\.?\d*)', text)
            
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
    def __init__(self, session, url, tokenizer, min_input=500, max_input=4000):
        self.session = session
        self.url = url
        self.tokenizer = tokenizer
        self.min_input = min_input
        self.max_input = max_input
        self.active_tasks = set()
        self.target_concurrency = 0
        self.running = False

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
    
    concurrency_sweep = [0, 10, 20, 30, 40, 50, 55, 60, 65, 70, 80, 90]
    
    # --- NEW: Probe Lengths Configuration ---
    probe_lengths = [50, 500, 1000, 2000, 4000] 
    
    data_points = [] 

    async with aiohttp.ClientSession() as session:
        load_manager = BackgroundLoadManager(session, args.url, tokenizer)
        bg_task = load_manager.start()

        for conc in concurrency_sweep:
            print(f"\n=============================================")
            print(f"SETTING BACKGROUND CONCURRENCY: {conc}")
            print(f"=============================================")
            load_manager.set_concurrency(conc)
            
            print(f"  > Waiting {args.warmup}s for queue to stabilize...")
            await asyncio.sleep(args.warmup)

            # --- Inner Loop: Iterate through requested probe lengths ---
            for p_len in probe_lengths:
                print(f"  > [Conc:{conc}] Testing Probe Input Length: {p_len}")
                
                # Generate exact length prompt
                # We repeat the seed text enough times to ensure we can slice p_len tokens
                seed_text = "test " * (p_len + 50) 
                encoded_ids = tokenizer.encode(seed_text, add_special_tokens=False)
                probe_prompt = tokenizer.decode(encoded_ids[:p_len], skip_special_tokens=True)

                for i in range(args.probes):
                    # 1. GET METRICS SNAPSHOT
                    n_run, n_wait, kv_usage = await get_vllm_metrics(session, args.metrics_url)
                    
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
                                        break # Got TTFT
                    except Exception as e:
                        print(f"Probe failed: {e}")

                    # 3. LOG DATA
                    if ttft is not None:
                        ttft_ms = ttft * 1000
                        print(f"    [Probe {i} | Len {p_len}] Wait:{n_wait:3.0f} | Run:{n_run:3.0f} | TTFT: {ttft_ms:.2f}ms")
                        
                        data_points.append({
                            "Probe_Length": p_len, # <--- Added this field
                            "Running_Reqs": n_run,
                            "Waiting_Reqs": n_wait,
                            "KV_Cache_Usage": kv_usage,
                            "TTFT_ms": ttft_ms
                        })
                    
                    await asyncio.sleep(random.uniform(0.2, 0.5))

        load_manager.stop()
        await bg_task

    # Save to CSV
    df = pd.DataFrame(data_points)
    csv_filename = "queue_v_ttft_multi_len.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nData saved to {csv_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1/completions")
    parser.add_argument("--metrics-url", type=str, default="http://localhost:8002/metrics")
    parser.add_argument("--model-path", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--probes", type=int, default=10, help="Probes per config") # Reduced default probes slightly to save time
    parser.add_argument("--warmup", type=int, default=8, help="Warmup seconds")
    
    args = parser.parse_args()
    asyncio.run(run_experiment(args))