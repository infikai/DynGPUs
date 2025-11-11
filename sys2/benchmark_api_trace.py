import argparse
import asyncio
import json
import os
import random
import time
# --- MODIFIED: Added asdict ---
from dataclasses import dataclass, field, asdict
from typing import List

import aiohttp
import numpy as np

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
    # Note: Sorting is now done after the optional downsampling step
    return requests


async def benchmark(
    api_url: str,
    model_name: str,
    requests: List[Request],
    duration: int,
):
    """Main benchmark function to send HTTP requests based on trace file."""
    
    async def process_request(session: aiohttp.ClientSession, request: Request) -> RequestResult:
        """Coroutine to send one HTTP request and capture detailed timestamps."""
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

        results = await asyncio.gather(*tasks)
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


# --- NEW FUNCTION ---
def save_results_to_file(results: List[RequestResult], filename: str):
    """Saves the detailed results list to a JSONL file."""
    print(f"\nSaving detailed results for {len(results)} requests to {filename}...")
    count = 0
    with open(filename, "w") as f:
        for result in results:
            try:
                # Convert the RequestResult dataclass to a dictionary
                result_dict = asdict(result)
                # Dump the dictionary as a JSON string
                json_line = json.dumps(result_dict)
                # Write the JSON string as a new line in the file
                f.write(json_line + "\n")
                count += 1
            except Exception as e:
                # Log a warning if a single result fails to serialize
                print(f"Warning: Failed to serialize result for request {result.request_id}: {e}")
    print(f"Successfully saved {count} individual request results.")

def save_summary_to_file(summary_report: str, filename: str):
    """Saves the summary report string to a text file."""
    print(f"\nSaving summary report to {filename}...")
    try:
        with open(filename, "w") as f:
            f.write(summary_report)
        print("Summary report saved successfully.")
    except IOError as e:
        print(f"Error: Failed to write summary file {filename}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A client for benchmarking vLLM API server with a trace file.")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--endpoint", type=str, default="/v1/completions")
    parser.add_argument("--model-name", type=str, default="NousResearch/Meta-Llama-3-8B-Instruct", help="The name of the model being served.")
    parser.add_argument("--trace-file", type=str, required=True, help="Path to the JSONL trace file.")
    parser.add_argument("--duration", type=int, default=None, help="Benchmark duration in seconds. If set, stops sending new requests after this time.")
    
    # --- NEW DOWNSAMPLING ARGUMENT ---
    parser.add_argument(
        "--downsample-factor",
        type=float,
        default=None,
        help="A factor between 0.0 and 1.0 to randomly sample a fraction of requests from the trace file."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42, # A default seed ensures it's reproducible by default
        help="Random seed for downsampling to ensure reproducibility."
    )
    
    # --- NEW OUTPUT FILE ARGUMENT ---
    parser.add_argument(
        "--output-file",
        type=str,
        default="./result.jsonl",
        help="Path to a JSONL file to save detailed results for each request."
    )

    parser.add_argument(
        "--summary-file",
        type=str,
        default="./summary.txt",
        help="Path to a TXT file to save the final summary report."
    )
    
    args = parser.parse_args()

    api_url = f"http://{args.host}:{args.port}{args.endpoint}"
    
    requests = load_trace_file(args.trace_file)
    
    # --- NEW DOWNSAMPLING LOGIC ---
    if args.downsample_factor:
        if not 0.0 < args.downsample_factor <= 1.0:
            raise ValueError("Downsample factor must be between 0.0 and 1.0.")
        
        # --- NEW: Set the random seed before sampling ---
        print(f"Using random seed: {args.seed}")
        random.seed(args.seed)
        
        original_count = len(requests)
        num_to_sample = int(original_count * args.downsample_factor)
        
        print(f"Downsampling trace from {original_count} to {num_to_sample} requests ({args.downsample_factor:.1%})...")
        requests = random.sample(requests, num_to_sample)

    # Re-sort requests by timestamp after any potential sampling
    requests.sort(key=lambda r: r.timestamp)
    
    start_time = time.perf_counter()
    results = asyncio.run(benchmark(api_url, args.model_name, requests, args.duration))
    end_time = time.perf_counter()
    
    actual_duration = end_time - start_time
    
    # --- NEW: Save results to file if an output file is specified ---
    if args.output_file:
        save_results_to_file(results, args.output_file)
    
    summary_report = calculate_metrics(requests, results, actual_duration)
    
    # Print summary to console
    print(summary_report)
    
    # Save summary to file if specified
    if args.summary_file:
        save_summary_to_file(summary_report, args.summary_file)