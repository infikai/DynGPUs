import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import List

import aiohttp
import numpy as np

# --- Configuration ---
# This file is used to communicate with the separate autoscaler script.
CONCURRENCY_FILE_PATH = "/mydata/Data/DynGPUs/sys/vllm_concurrency.txt"


@dataclass
class Request:
    """Represents a single request from the trace file."""
    timestamp: float
    prompt: str
    output_len: int
    request_id: int


@dataclass
class RequestResult:
    """Stores the detailed results and timestamps for a single request."""
    request_id: int
    success: bool
    output_len: int
    start_time: float = 0.0
    first_token_time: float = 0.0
    end_time: float = 0.0
    token_timestamps: List[float] = field(default_factory=list)


def load_trace_file(filepath: str) -> List[Request]:
    """Loads and parses the trace file, creating a list of requests."""
    requests = []
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                requests.append(
                    Request(
                        timestamp=float(data["timestamp"]),
                        prompt=data["prompt"],
                        output_len=data["output_len"],
                        request_id=i,
                    )
                )
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Skipping malformed line {i+1} in trace file: {e}")
    requests.sort(key=lambda r: r.timestamp)
    return requests


async def monitor_concurrency(tasks: List[asyncio.Task], results: List):
    """A background task to WRITE the number of active requests to a file every second."""
    while True:
        try:
            active_tasks = len(tasks) - len(results)
            # Write the current number to the shared file for the autoscaler to read
            with open(CONCURRENCY_FILE_PATH, "w") as f:
                f.write(str(active_tasks))
        except Exception as e:
            print(f"Monitor error: {e}")
            
        # Update the concurrency value every 1 second
        await asyncio.sleep(1)


async def benchmark(
    api_url: str,
    model_name: str,
    requests: List[Request],
    duration: int,
):
    """Main benchmark function to send HTTP requests based on trace file."""
    
    async def process_request(session: aiohttp.ClientSession, request: Request) -> RequestResult:
        # This inner function remains the same as before
        result = RequestResult(request_id=request.request_id, success=False, output_len=0)
        
        payload = {
            "model": model_name,
            "prompt": request.prompt,
            "max_tokens": request.output_len,
            "stream": True,
        }
        
        result.start_time = time.perf_counter()
        generated_tokens = 0
        
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status != 200:
                    print(f"Error: Request {request.request_id} failed with status {response.status}")
                    result.end_time = time.perf_counter()
                    return result
                
                first_token_received = False
                async for chunk in response.content.iter_any():
                    token_time = time.perf_counter()
                    if not first_token_received:
                        result.first_token_time = token_time
                        first_token_received = True
                    result.token_timestamps.append(token_time)
                    if chunk.strip():
                        generated_tokens += 1

            result.end_time = time.perf_counter()
            result.output_len = generated_tokens
            result.success = True
            
        except aiohttp.ClientError as e:
            print(f"Error processing request {request.request_id}: {e}")
            result.end_time = time.perf_counter()

        return result

    # Main benchmark loop
    conn = aiohttp.TCPConnector(limit=None)
    async with aiohttp.ClientSession(connector=conn) as session:
        benchmark_start_time = time.perf_counter()
        tasks: List[asyncio.Task] = []
        results: List[RequestResult] = []

        # Start the concurrency monitoring task
        monitor_task = asyncio.create_task(
            monitor_concurrency(tasks, results)
        )

        print("Dispatching requests...")
        for request in requests:
            if duration is not None and (time.perf_counter() - benchmark_start_time) >= duration:
                print(f"\nBenchmark duration of {duration}s reached. No more requests will be sent.")
                break

            current_time = time.perf_counter() - benchmark_start_time
            time_to_wait = request.timestamp - current_time
            if time_to_wait > 0:
                await asyncio.sleep(time_to_wait)
                
            task = asyncio.create_task(process_request(session, request))
            tasks.append(task)
        print("All requests have been dispatched. Waiting for completion...")

        # --- CHANGE: Use asyncio.as_completed to get results in real-time ---
        # This is the key fix. We now loop over tasks as they finish.
        # The 'results' list will now grow incrementally, giving the monitor
        # an accurate, real-time count of completed tasks.
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)

        # Stop the monitoring task
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass # Expected cancellation

        return results


def calculate_metrics(
    requests: List[Request],
    results: List[RequestResult],
    duration: float,
):
    """Calculates and prints performance metrics."""
    completed_requests = sum(1 for r in results if r.success)
    total_output_tokens = sum(r.output_len for r in results if r.success)

    print("\n" + "="*50)
    print("=============== Benchmark Summary ================")
    print("="*50)
    print(f"Total time: {duration:.2f} s")
    print(f"Total requests processed: {completed_requests} / {len(results)}")
    print(f"Throughput (requests/sec): {completed_requests / duration:.2f}")
    print(f"Throughput (output tokens/sec): {total_output_tokens / duration:.2f}")

    ttfts, tpots, itls = [], [], []
    for res in results:
        if not res.success or res.output_len == 0:
            continue
        
        ttfts.append(res.first_token_time - res.start_time)
        if res.output_len > 1:
            tpot = (res.end_time - res.first_token_time) / (res.output_len - 1)
            tpots.append(tpot)
            inter_token_latencies = np.diff(res.token_timestamps)
            itls.extend(inter_token_latencies.tolist())

    def print_latency_stats(name, latencies_sec):
        if not latencies_sec: return
        latencies_ms = np.array(latencies_sec) * 1000
        print(f"Mean {name} (ms):   {np.mean(latencies_ms):.2f}")
        print(f"Median {name} (ms): {np.median(latencies_ms):.2f}")
        print(f"P99 {name} (ms):    {np.percentile(latencies_ms, 99):.2f}")

    print("\n" + "-"*15 + "Time to First Token" + "-"*15)
    print_latency_stats("TTFT", ttfts)
    print("\n" + "-----Time per Output Token (excl. 1st token)------")
    print_latency_stats("TPOT", tpots)
    print("\n" + "-"*15 + "Inter-token Latency" + "-"*15)
    print_latency_stats("ITL", itls)
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A client for benchmarking vLLM API server with a trace file.")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--endpoint", type=str, default="/v1/completions")
    parser.add_argument("--model-name", type=str, required=True, help="The name of the model being served.")
    parser.add_argument("--trace-file", type=str, required=True, help="Path to the JSONL trace file.")
    parser.add_argument("--duration", type=int, default=None, help="Benchmark duration in seconds. If set, stops sending new requests after this time.")
    args = parser.parse_args()

    api_url = f"http://{args.host}:{args.port}{args.endpoint}"
    
    requests = load_trace_file(args.trace_file)
    
    start_time = time.perf_counter()
    results = asyncio.run(benchmark(api_url, args.model_name, requests, args.duration))
    end_time = time.perf_counter()
    
    actual_duration = end_time - start_time
    calculate_metrics(requests, results, actual_duration)