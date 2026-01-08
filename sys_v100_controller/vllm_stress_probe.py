import argparse
import asyncio
import aiohttp
import time
import re
import random
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Optional
import json

# --- Configuration ---
METRICS_POLL_INTERVAL = 0.5  # Check server status every 0.5s
STABILITY_PAUSE = 2.0        # Wait 2s after reaching target before probing
SATURATION_TIMEOUT = 10      # Timeout if server is stuck
BG_MIN_TOKENS = 128          # Minimum random output length
BG_MAX_TOKENS = 2048         # Maximum random output length
PROBES_PER_TEST = 10         # <--- NEW: 10 probes per data point
INTER_TEST_DELAY = 10.0      # <--- NEW: 10s wait between experiments

@dataclass
class ProbeResult:
    target_load: int
    actual_load: float
    probe_type: str 
    input_len: int
    ttft_avg_ms: float
    ttft_std_ms: float
    raw_ttfts: List[float]

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
        # print(f"  [Controller] Ensuring load is at {target_running}...")
        
        while self._keep_running:
            await self.get_server_state()
            
            # Clean up finished python tasks from our list
            self._background_tasks = [t for t in self._background_tasks if not t.done()]
            
            deficit = target_running - int(self.current_running)
            
            # STOP condition: We match the target
            if deficit <= 0:
                break

            # SAFETY: Detect Saturation
            if self.current_waiting > 2 and self.current_running < target_running:
                 # print(f"  [Controller] Saturation detected! Stuck at {self.current_running}.")
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

async def run_test_block(
    controller: ServerController, 
    api_url: str, 
    model_name: str, 
    target_load: int, 
    input_len: int,
    probe_type: str
) -> ProbeResult:
    """
    Runs the 10-probe sequence for a specific configuration.
    """
    latencies = []
    load_during_probes = []
    
    print(f"  -> Starting Test: Load={target_load}, InputLen={input_len}")

    for i in range(PROBES_PER_TEST):
        # 1. Ensure state is correct before EVERY probe
        await controller.maintain_load(target_load)
        
        # 2. Wait a tiny bit for stability
        await asyncio.sleep(0.5)
        
        # 3. Record actual load just before probe
        load_during_probes.append(controller.current_running)
        
        # 4. Probe
        ttft = await send_probe(api_url, model_name, input_len)
        if ttft > 0:
            latencies.append(ttft)
        
        # 5. Small gap between probes
        await asyncio.sleep(0.2)

    if not latencies:
        return None

    avg_ttft = np.mean(latencies)
    std_ttft = np.std(latencies)
    avg_load = np.mean(load_during_probes)
    
    print(f"     Results: Avg TTFT={avg_ttft:.2f}ms (Std={std_ttft:.2f}) | Avg Load={avg_load:.1f}")
    
    return ProbeResult(
        target_load=target_load,
        actual_load=avg_load,
        probe_type=probe_type,
        input_len=input_len,
        ttft_avg_ms=avg_ttft,
        ttft_std_ms=std_ttft,
        raw_ttfts=latencies
    )

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=8000)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="averaged_results.jsonl")
    args = parser.parse_args()

    metrics_url = f"http://{args.host}:{args.port}/metrics"
    api_url = f"http://{args.host}:{args.port}/v1/completions"

    controller = ServerController(metrics_url, args.model)
    await controller.start()

    all_results = []

    try:
        # --- Experiment 1: TTFT vs Running Requests ---
        print(f"\n=== Experiment 1: TTFT vs Running Requests ({PROBES_PER_TEST} probes avg) ===")
        targets = [5, 10, 20, 40, 60] 
        
        for target in targets:
            res = await run_test_block(controller, api_url, args.model, target, 100, "concurrency")
            if res: all_results.append(res)
            
            print(f"     Cooling down for {INTER_TEST_DELAY}s...")
            await asyncio.sleep(INTER_TEST_DELAY)

        # Reset Controller
        await controller.stop()
        controller = ServerController(metrics_url, args.model)
        await controller.start()
        
        # --- Experiment 2: TTFT vs Input Length ---
        print(f"\n=== Experiment 2: TTFT vs Input Length ({PROBES_PER_TEST} probes avg) ===")
        FIXED_LOAD = 10 
        input_lengths = [50, 500, 1000, 2000, 4000]
        
        for length in input_lengths:
            res = await run_test_block(controller, api_url, args.model, FIXED_LOAD, length, "input_len")
            if res: all_results.append(res)
            
            print(f"     Cooling down for {INTER_TEST_DELAY}s...")
            await asyncio.sleep(INTER_TEST_DELAY)

    finally:
        await controller.stop()
        
        # Save results
        with open(args.output, "w") as f:
            for r in all_results:
                f.write(json.dumps(asdict(r)) + "\n")
        print(f"\nSaved averaged results to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())