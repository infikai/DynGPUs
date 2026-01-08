import argparse
import asyncio
import aiohttp
import time
import re
import random
from dataclasses import dataclass
from typing import List, Optional

# --- Configuration ---
METRICS_POLL_INTERVAL = 0.5  # Check server status every 0.5s
STABILITY_PAUSE = 1.0        # Wait 2s after reaching target before probing
SATURATION_TIMEOUT = 10      # Timeout if server is stuck
BG_MIN_TOKENS = 128          # Minimum random output length
BG_MAX_TOKENS = 2048         # Maximum random output length

@dataclass
class ProbeResult:
    target_load: int
    actual_load: int
    probe_type: str 
    input_len: int
    ttft_ms: float

class ServerController:
    """
    Watches vLLM metrics and maintains a target number of running requests
    by replacing finished requests with new random-length ones.
    """
    def __init__(self, metrics_url: str, model_name: str):
        self.metrics_url = metrics_url
        self.model_name = model_name
        self.current_running = 0.0
        self.current_waiting = 0.0
        self._session = None
        self._background_tasks = []
        self._keep_running = True

    async def start(self):
        self._session = aiohttp.ClientSession()

    async def stop(self):
        self._keep_running = False
        # Cancel all background tasks
        for t in self._background_tasks:
            t.cancel()
        if self._session:
            await self._session.close()

    async def get_server_state(self):
        """Fetches and parses vLLM metrics."""
        try:
            async with self._session.get(self.metrics_url) as response:
                if response.status != 200:
                    return
                text = await response.text()
                
                # Parse Prometheus format
                running_match = re.search(r'vllm:num_requests_running.*?\}\s+([0-9\.]+)', text)
                waiting_match = re.search(r'vllm:num_requests_waiting.*?\}\s+([0-9\.]+)', text)
                
                if running_match:
                    self.current_running = float(running_match.group(1))
                if waiting_match:
                    self.current_waiting = float(waiting_match.group(1))
                    
        except Exception as e:
            print(f"Warning: Failed to fetch metrics: {e}")

    async def maintain_load(self, target_running: int):
        """
        The Control Loop:
        Refills the server with requests if random-length tasks finish.
        """
        print(f"  [Controller] Stabilizing load at {target_running} running requests...")
        
        while self._keep_running:
            await self.get_server_state()
            
            # Clean up finished python tasks from our list
            self._background_tasks = [t for t in self._background_tasks if not t.done()]
            
            deficit = target_running - int(self.current_running)
            
            # STOP condition: We match the target
            # Note: We continue monitoring if we are in "maintenance mode", 
            # but for the probe logic, we usually just want to reach the state once.
            if deficit <= 0:
                # If we are just holding state, we break here. 
                # (Ideally, you'd run this in a separate async task to hold it indefinitely, 
                # but for this script's flow, we just top it up before probing).
                break

            # SAFETY: Detect Saturation
            if self.current_waiting > 2 and self.current_running < target_running:
                 print(f"  [Controller] Saturation detected! Stuck at {self.current_running}. (Queue: {self.current_waiting})")
                 break

            # Action: Launch new background workers to fill deficit
            if deficit > 0:
                batch = min(deficit, 5) 
                for _ in range(batch):
                    t = asyncio.create_task(self._background_worker())
                    self._background_tasks.append(t)
                await asyncio.sleep(0.5) # Allow registration
            
            await asyncio.sleep(METRICS_POLL_INTERVAL)

    async def _background_worker(self):
        """Generates a request with RANDOM output length."""
        try:
            # Randomize output length
            req_output_len = random.randint(BG_MIN_TOKENS, BG_MAX_TOKENS)
            
            payload = {
                "model": self.model_name,
                "prompt": "Repeat this: " + "test " * 50,
                "max_tokens": req_output_len, # <--- RANDOMIZED HERE
                "stream": True,
                "ignore_eos": True
            }
            
            # Use a ephemeral session for workers
            async with aiohttp.ClientSession() as session:
                endpoint = self.metrics_url.replace("/metrics", "/v1/completions")
                async with session.post(endpoint, json=payload) as resp:
                    async for _ in resp.content:
                        if not self._keep_running: break
                        pass # Consume stream until done
        except Exception:
            pass

async def send_probe(api_url: str, model_name: str, input_len: int) -> float:
    """Sends a single probe request and returns TTFT (ms)."""
    prompt = "test " * input_len
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": 10,
        "stream": True
    }
    
    start = time.perf_counter()
    first_token = None
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(api_url, json=payload) as resp:
                async for line in resp.content:
                    if line.startswith(b"data: "):
                        first_token = time.perf_counter()
                        break 
        except Exception as e:
            print(f"Probe failed: {e}")
            return -1

    if first_token:
        return (first_token - start) * 1000
    return -1

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=8000)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    metrics_url = f"http://{args.host}:{args.port}/metrics"
    api_url = f"http://{args.host}:{args.port}/v1/completions"

    controller = ServerController(metrics_url, args.model)
    await controller.start()

    try:
        # --- Experiment 1: TTFT vs Running Requests ---
        print("\n=== Experiment 1: TTFT vs Running Requests (Random Bg Lengths) ===")
        targets = [5, 10, 20, 40, 60] 
        
        for target in targets:
            # 1. Top up the load
            await controller.maintain_load(target)
            
            # 2. Wait a moment (some might finish, new ones might start)
            await asyncio.sleep(STABILITY_PAUSE)
            
            # 3. Top up AGAIN right before probing to ensure we are close to target
            await controller.maintain_load(target)
            
            actual = controller.current_running
            ttft = await send_probe(api_url, args.model, input_len=100)
            
            print(f"  -> Target: {target} | Actual: {actual} | TTFT: {ttft:.2f} ms")

        # Reset
        await controller.stop()
        controller = ServerController(metrics_url, args.model)
        await controller.start()
        
        # --- Experiment 2: TTFT vs Input Length ---
        print("\n=== Experiment 2: TTFT vs Input Length (Fixed Random-Len Load) ===")
        FIXED_LOAD = 10 
        input_lengths = [50, 500, 1000, 2000]
        
        for length in input_lengths:
            # Ensure load is maintained before every single probe
            await controller.maintain_load(FIXED_LOAD)
            await asyncio.sleep(1.0)
            
            actual = controller.current_running
            ttft = await send_probe(api_url, args.model, input_len=length)
            
            print(f"  -> Length: {length} | Actual Load: {actual} | TTFT: {ttft:.2f} ms")

    finally:
        await controller.stop()

if __name__ == "__main__":
    asyncio.run(main())