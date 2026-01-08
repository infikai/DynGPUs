import asyncio
import aiohttp
import argparse
import time
import random
import numpy as np
from transformers import AutoTokenizer

# ==============================================================================
# 1. HELPER: Exact Token Prompt Generator (Still used for the PROBE)
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
# 2. CLASS: Background Load Manager (Dynamic Inputs)
# ==============================================================================
class BackgroundLoadManager:
    """
    Maintains concurrency with DYNAMIC input lengths to simulate real traffic.
    """
    def __init__(self, session, url, tokenizer, min_input=128, max_input=1024):
        self.session = session
        self.url = url
        self.tokenizer = tokenizer
        self.min_input = min_input
        self.max_input = max_input
        self.active_tasks = set()
        self.target_concurrency = 0
        self.running = False

        # OPTIMIZATION: Pre-calculate a huge buffer of token IDs.
        # We will slice this buffer later instead of re-encoding every time.
        print("  [LoadManager] Pre-calculating token buffer...")
        base_text = "The quick brown fox jumps over the lazy dog. "
        base_ids = tokenizer.encode(base_text, add_special_tokens=False)
        self.token_buffer = []
        
        # Ensure buffer is large enough for the maximum requested input
        while len(self.token_buffer) < max_input + 50: # +50 for safety
            self.token_buffer.extend(base_ids)
            
    async def _worker(self):
        """Continuous loop to maintain concurrency."""
        while self.running:
            current_load = len(self.active_tasks)
            
            if current_load < self.target_concurrency:
                deficit = self.target_concurrency - current_load
                for _ in range(deficit):
                    task = asyncio.create_task(self._send_dynamic_background_request())
                    self.active_tasks.add(task)
                    task.add_done_callback(self.active_tasks.discard)
            
            await asyncio.sleep(0.05)

    async def _send_dynamic_background_request(self):
        """Generates a random input length prompt and fires it."""
        try:
            # 1. Determine Random Input Length
            input_len = random.randint(self.min_input, self.max_input)
            
            # 2. Slice the pre-calculated buffer (Fast!)
            # We don't need to re-encode "The quick brown..." every time.
            slice_ids = self.token_buffer[:input_len]
            
            # 3. Decode to string (API expects text)
            prompt_text = self.tokenizer.decode(slice_ids, skip_special_tokens=True)
            
            # 4. Random Output Length
            output_len = random.randint(128, 512)
            
            payload = {
                "prompt": prompt_text,
                "max_tokens": output_len,
                "temperature": 0.9,
                "stream": False # Background noise doesn't need streaming
            }
            
            async with self.session.post(self.url, json=payload) as resp:
                # We must consume the response to free the connection
                await resp.read() 
                
        except Exception:
            pass # Ignore errors in background noise

    def set_concurrency(self, n):
        self.target_concurrency = n

    def start(self):
        self.running = True
        return asyncio.create_task(self._worker())

    def stop(self):
        self.running = False

# ==============================================================================
# 3. CORE: Probe Logic
# ==============================================================================
async def send_probe(session, url, prompt, request_id):
    """Sends a probe request and measures TTFT."""
    payload = {
        "prompt": prompt,
        "max_tokens": 10,
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
                    # Stop reading to save client CPU, server will finish generation
                    break 
    except Exception as e:
        print(f"    [Probe {request_id}] Error: {e}")
        return None
        
    return ttft

# ==============================================================================
# 4. MAIN EXPERIMENT LOOP
# ==============================================================================
async def run_experiment(args):
    print(f"Loading Tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # --- CONFIGURATION ---
    concurrency_levels = [0, 5, 10, 20, 30, 40] # Add more as needed
    token_lengths = [50, 100, 500, 1000, 2000, 4000]     # Add more as needed
    
    # Background Noise Config
    bg_min_input = 200
    bg_max_input = 2000
    # ---------------------

    final_results = [] 

    async with aiohttp.ClientSession() as session:
        # Initialize Load Manager with Dynamic Input Range
        load_manager = BackgroundLoadManager(
            session, args.url, tokenizer, 
            min_input=bg_min_input, max_input=bg_max_input
        )
        bg_task = load_manager.start()

        for concurrency in concurrency_levels:
            print(f"\n========================================")
            print(f"SETTING BACKGROUND CONCURRENCY: {concurrency}")
            print(f"  (Background Inputs: Random {bg_min_input}-{bg_max_input} tokens)")
            print(f"========================================")
            
            load_manager.set_concurrency(concurrency)
            
            # Warmup
            if concurrency > 0:
                print(f"  > Warming up for {args.warmup}s...")
                while len(load_manager.active_tasks) < concurrency:
                    await asyncio.sleep(0.1)
                await asyncio.sleep(args.warmup)

            # Test Loop
            for n_tokens in token_lengths:
                print(f"  > Testing Probe Input Tokens: {n_tokens}")
                
                probe_prompt = generate_fixed_prompt(tokenizer, n_tokens)
                latencies = []
                
                for i in range(args.probes):
                    ttft = await send_probe(session, args.url, probe_prompt, i)
                    if ttft:
                        latencies.append(ttft)
                    await asyncio.sleep(0.5)

                if latencies:
                    avg_ttft = np.mean(latencies)
                    print(f"    -> Avg TTFT: {avg_ttft*1000:.2f} ms")
                    final_results.append((concurrency, n_tokens, avg_ttft))
                else:
                    final_results.append((concurrency, n_tokens, 0.0))

        load_manager.stop()
        await bg_task

    print("\n\n=== FINAL RESULTS SUMMARY (Dynamic Bg Input) ===")
    print(f"{'Concurrent Reqs':<20} | {'Probe Tokens':<15} | {'Avg TTFT (ms)':<15}")
    print("-" * 60)
    for res in final_results:
        print(f"{res[0]:<20} | {res[1]:<15} | {res[2]*1000:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1/completions")
    parser.add_argument("--model-path", type=str, required=True, help="HF Model ID/Path")
    parser.add_argument("--probes", type=int, default=10, help="Probes per config")
    parser.add_argument("--warmup", type=int, default=10, help="Seconds to wait after changing load")
    
    args = parser.parse_args()
    asyncio.run(run_experiment(args))