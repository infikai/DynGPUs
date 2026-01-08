import asyncio
import aiohttp
import argparse
import time
import random
import numpy as np
from transformers import AutoTokenizer

# ==============================================================================
# 1. HELPER: Exact Token Prompt Generator
# ==============================================================================
def generate_fixed_prompt(tokenizer, target_tokens, base_text="The quick brown fox jumps over the lazy dog. "):
    test_ids = tokenizer.encode("")
    current_ids = list(test_ids)
    base_ids = tokenizer.encode(base_text, add_special_tokens=False)
    
    while len(current_ids) < target_tokens:
        current_ids.extend(base_ids)
        
    final_ids = current_ids[:target_tokens]
    return tokenizer.decode(final_ids, skip_special_tokens=True)

# ==============================================================================
# 2. CLASS: Background Load Manager
# ==============================================================================
class BackgroundLoadManager:
    """
    Maintains a specific number of concurrent requests to stress the server.
    """
    def __init__(self, session, url, tokenizer, prompt_text):
        self.session = session
        self.url = url
        self.tokenizer = tokenizer
        self.prompt_text = prompt_text
        self.active_tasks = set()
        self.target_concurrency = 0
        self.running = False
        self.stats_lock = asyncio.Lock()

    async def _worker(self):
        """Continuous loop to maintain concurrency."""
        while self.running:
            # Check if we need to spawn more requests
            current_load = len(self.active_tasks)
            
            if current_load < self.target_concurrency:
                # Spawn needed amount (or just 1 to avoid thundering herd)
                deficit = self.target_concurrency - current_load
                for _ in range(deficit):
                    task = asyncio.create_task(self._send_background_request())
                    self.active_tasks.add(task)
                    task.add_done_callback(self.active_tasks.discard)
            
            # Sleep briefly to yield control
            await asyncio.sleep(0.05)

    async def _send_background_request(self):
        """Sends a 'noise' request with random output length."""
        try:
            # Random output length for background noise (as requested)
            out_len = random.randint(128, 512)
            
            payload = {
                "prompt": self.prompt_text,
                "max_tokens": out_len,
                "temperature": 0.9,
                "stream": False # We don't care about background TTFT
            }
            async with self.session.post(self.url, json=payload) as resp:
                await resp.read() # Consume response
        except Exception:
            pass # Ignore background errors

    def set_concurrency(self, n):
        self.target_concurrency = n

    def start(self):
        self.running = True
        return asyncio.create_task(self._worker())

    def stop(self):
        self.running = False
        # Ideally, we might cancel active tasks here, but letting them drain is safer for the server

# ==============================================================================
# 3. CORE: Probe Logic
# ==============================================================================
async def send_probe(session, url, prompt, request_id):
    """Sends a probe request and measures TTFT."""
    payload = {
        "prompt": prompt,
        "max_tokens": 10, # Keep output short for probes, we only want TTFT
        "temperature": 0.0,
        "stream": True
    }
    
    start = time.time()
    ttft = None
    
    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                print(f"    [Probe {request_id}] Failed: {response.status}")
                return None
            
            async for chunk in response.content:
                if ttft is None:
                    ttft = time.time() - start
                    # We can abort stream early since we only need TTFT
                    # But be careful: aborting might cause server side errors. 
                    # For benchmarks, safer to read whole stream or set max_tokens=1.
    except Exception as e:
        print(f"    [Probe {request_id}] Error: {e}")
        return None
        
    return ttft

# ==============================================================================
# 4. MAIN EXPERIMENT LOOP
# ==============================================================================
async def run_experiment(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Define Sweep Parameters
    concurrency_levels = [0, 5, 10, 20] # Add more as needed
    token_lengths = [512, 1024, 2048]     # Add more as needed
    
    # Pre-generate a simple prompt for background noise (fixed length ~200 tokens)
    bg_prompt = generate_fixed_prompt(tokenizer, 200)

    final_results = [] # Stores tuple: (concurrency, input_len, avg_ttft)

    async with aiohttp.ClientSession() as session:
        # Initialize Load Manager
        load_manager = BackgroundLoadManager(session, args.url, tokenizer, bg_prompt)
        bg_task = load_manager.start()

        for concurrency in concurrency_levels:
            print(f"\n========================================")
            print(f"SETTING BACKGROUND CONCURRENCY: {concurrency}")
            print(f"========================================")
            
            # 1. Ramp up load
            load_manager.set_concurrency(concurrency)
            
            # 2. Wait for load to stabilize (Warmup)
            if concurrency > 0:
                print(f"  > Warming up for {args.warmup}s to reach steady state...")
                while len(load_manager.active_tasks) < concurrency:
                    await asyncio.sleep(0.1)
                await asyncio.sleep(args.warmup)

            # 3. Run Probes for each token length
            for n_tokens in token_lengths:
                print(f"  > Testing Input Tokens: {n_tokens}")
                
                # Generate Probe Prompt
                probe_prompt = generate_fixed_prompt(tokenizer, n_tokens)
                
                latencies = []
                for i in range(args.probes):
                    ttft = await send_probe(session, args.url, probe_prompt, i)
                    if ttft:
                        latencies.append(ttft)
                    await asyncio.sleep(0.5) # Gap between probes

                if latencies:
                    avg_ttft = np.mean(latencies)
                    print(f"    -> Avg TTFT: {avg_ttft*1000:.2f} ms")
                    final_results.append((concurrency, n_tokens, avg_ttft))
                else:
                    final_results.append((concurrency, n_tokens, 0.0))

        # Cleanup
        load_manager.stop()
        await bg_task

    # Output Table
    print("\n\n=== FINAL RESULTS SUMMARY ===")
    print(f"{'Concurrent Reqs':<20} | {'Input Tokens':<15} | {'Avg TTFT (ms)':<15}")
    print("-" * 60)
    for res in final_results:
        print(f"{res[0]:<20} | {res[1]:<15} | {res[2]*1000:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1/completions")
    parser.add_argument("--model-path", type=str, required=True, help="HF Model ID/Path")
    parser.add_argument("--probes", type=int, default=5, help="Probes per config")
    parser.add_argument("--warmup", type=int, default=5, help="Seconds to wait after changing load")
    
    args = parser.parse_args()
    
    asyncio.run(run_experiment(args))