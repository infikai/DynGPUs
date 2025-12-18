import argparse
import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Union

import aiohttp
import numpy as np


# --- Global Configuration ---
STAGE_DURATION_SECONDS = 300  # 5 minutes per stage
BENCHMARK_STAGES = {
    "Stage 1": 0.5,
    "Stage 2": 0.8,
    "Stage 3": 1,
    "Stage 4": 0.6,
    "Stage 5": 0.4,
    "Stage 6": 0.9,
    "Stage 7": 0.5,
    "Stage 8": 0.8,
    "Stage 9": 0.4,
}
REQUEST_READ_TIMEOUT_SECONDS = 600


# --- Data Classes ---

@dataclass
class Request:
    """Represents a single request, potentially loaded from trace."""
    timestamp: float # Original trace timestamp (now unused for dispatch)
    prompt: str
    output_len: int
    request_id: int
    stage: str = "synthetic" # Used to track which stage dispatched this request


@dataclass
class RequestResult:
    """Stores the detailed results and timestamps for a single request."""
    request_id: int
    success: bool
    output_len: int
    stage: str # Used for per-stage metric analysis
    start_time: float = 0.0
    first_token_time: float = 0.0
    end_time: float = 0.0
    token_timestamps: List[float] = field(default_factory=list)


# --- Utility Classes ---

class GlobalCounter:
    """Simple class to ensure unique, sequential request IDs."""
    def __init__(self):
        self._count = 0
    def next(self):
        self._count += 1
        return self._count


# --- Trace File Loader (Reintroduced) ---

def load_trace_file(filepath: str) -> List[Request]:
    """Loads and parses the trace file, creating a list of requests."""
    requests = []
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                # Note: We must create a new unique ID here, separate from the trace file's index (i)
                requests.append(
                    Request(
                        timestamp=float(data["timestamp"]),
                        prompt=data["prompt"],
                        output_len=data["output_len"],
                        request_id=i, # Use trace index as initial ID
                    )
                )
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Skipping malformed line {i+1} in trace file: {e}")
    return requests


# --- Core Executor ---

async def process_request(request: Request, api_url: str, model_name: str) -> RequestResult:
    """Coroutine to send one HTTP request to the single API target."""
    
    result = RequestResult(request_id=request.request_id, success=False, output_len=0, stage=request.stage)
    
    payload = {
        "model": model_name,
        "prompt": request.prompt,
        "max_tokens": request.output_len,
        "stream": True,
    }
    
    result.start_time = time.perf_counter()
    generated_tokens = 0
    
    # Fresh connection/session for every request to ensure distinct traffic
    connector = aiohttp.TCPConnector(force_close=True, enable_cleanup_closed=True)
    timeout = aiohttp.ClientTimeout(
        total=None, # No overall timeout limit
        connect=None, # No connection timeout limit
        sock_read=REQUEST_READ_TIMEOUT_SECONDS # Timeout between read chunks
    )
    
    try:
        async with aiohttp.ClientSession(connector=connector, cookie_jar=aiohttp.DummyCookieJar()) as session:
            # Explicitly set Connection: close header
            async with session.post(
                url=api_url, 
                json=payload, 
                headers={"Connection": "close"},
                timeout=timeout  # <--- HERE
            ) as response:
                if response.status != 200:
                    print(f"Error: Request {request.request_id} failed with status {response.status} from {api_url}")
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
        # print(f"Error processing request {request.request_id} to {api_url}: {e}")
        result.end_time = time.perf_counter()
    
    return result


# --- Master Dispatcher ---

def prepare_requests(all_requests: List[Request], stages: Dict[str, int], stage_duration: int, seed: int) -> List[Request]:
    """
    Samples requests from the trace file and assigns them to benchmark stages.
    """
    random.seed(seed)
    
    # 1. Calculate total requests needed across all stages
    total_requests_needed = int(sum(rps * stage_duration for rps in stages.values()))
    
    if total_requests_needed == 0:
        return []

    # 2. Check if trace file has enough data
    if len(all_requests) < total_requests_needed:
        print(f"Warning: Trace file only has {len(all_requests)} requests, but {total_requests_needed} are needed for the benchmark profile.")
        print("Repeating trace requests to meet the demand.")
        
        # Extend the request list by repeating it until we have enough
        num_repeats = (total_requests_needed + len(all_requests) - 1) // len(all_requests)
        extended_requests = all_requests * num_repeats
    else:
        extended_requests = all_requests
        
    # 3. Sample exactly the number of requests needed
    sampled_requests = random.sample(extended_requests, total_requests_needed)
    
    # 4. Partition and assign stage metadata
    prepared_requests: List[Request] = []
    global_counter = GlobalCounter()
    current_index = 0
    
    for stage_name, rps in stages.items():
        requests_in_stage = int(rps * stage_duration)
        
        # Take the next batch of sampled requests for this stage
        stage_requests = sampled_requests[current_index : current_index + requests_in_stage]
        
        for req in stage_requests:
            # Create a *copy* to assign the new stage and a unique ID
            new_req = Request(
                timestamp=req.timestamp,
                prompt=req.prompt,
                output_len=req.output_len,
                request_id=global_counter.next(), # Assign new sequential ID
                stage=stage_name
            )
            prepared_requests.append(new_req)
            
        current_index += requests_in_stage
        
    # The list is prepared but not sorted by timestamp (it's random), which is fine 
    # since dispatch is controlled by RPS delay, not timestamp.
    return prepared_requests


async def benchmark(
    api_url: str,
    model_name: str,
    stages: Dict[str, int],
    stage_duration: int,
    prepared_requests: List[Request]
):
    """Dispatches requests based on staged RPS, using content from the prepared trace requests."""
    
    all_tasks: List[asyncio.Task] = []
    benchmark_start_time = time.perf_counter()
    request_iterator = iter(prepared_requests)

    for stage_name, rps in stages.items():
        print(f"\n--- Starting Stage: {stage_name} (RPS: {rps}) for {stage_duration}s ---")
        stage_start_time = time.perf_counter()
        
        target_delay = 1.0 / rps if rps > 0 else float('inf')
        total_requests_to_dispatch = int(rps * stage_duration)
        
        for i in range(total_requests_to_dispatch):
            
            # --- Timing Control ---
            elapsed = time.perf_counter() - stage_start_time
            expected_next_dispatch_time = (i * target_delay)
            time_to_wait = expected_next_dispatch_time - elapsed
            if time_to_wait > 0:
                await asyncio.sleep(time_to_wait)
            
            # --- Dispatch ---
            try:
                # Get the next prepared request from the iterator
                request = next(request_iterator) 
            except StopIteration:
                print("Error: Iterator ran out of requests prematurely. Stopping benchmark.")
                break # Break out of the stage loop

            # Create and append the task
            task = asyncio.create_task(process_request(request, api_url, model_name))
            all_tasks.append(task)
            
        print(f"Stage {stage_name} dispatched {total_requests_to_dispatch} requests.")
        
        # Wait for the stage duration to pass (allows remaining tasks to run in the background)
        stage_runtime = time.perf_counter() - stage_start_time
        time_to_wait_after_dispatch = stage_duration - stage_runtime
        if time_to_wait_after_dispatch > 0:
            await asyncio.sleep(time_to_wait_after_dispatch)
        
        if total_requests_to_dispatch > 0:
            actual_stage_rps = total_requests_to_dispatch / stage_runtime
            print(f"  Actual Dispatch RPS during stage: {actual_stage_rps:.2f}")


    print("\nAll stages dispatched. Waiting for all requests to complete...")
    results = await asyncio.gather(*all_tasks)
    
    return results


# --- Utilities (saving and metrics) ---

def save_results_to_file(results: List[RequestResult], filename: str):
    """Saves the detailed results list to a JSONL file."""
    print(f"\nSaving detailed results for {len(results)} requests to {filename}...")
    count = 0
    with open(filename, "w") as f:
        for result in results:
            try:
                result_dict = asdict(result)
                json_line = json.dumps(result_dict)
                f.write(json_line + "\n")
                count += 1
            except Exception as e:
                print(f"Warning: Failed to serialize result for request {result.request_id}: {e}")
    print(f"Successfully saved {count} individual request results.")


def calculate_metrics(
    results: List[RequestResult],
    duration: float,
):
    """Calculates and prints performance metrics."""
    completed_requests = sum(1 for r in results if r.success)
    total_output_tokens = sum(r.output_len for r in results if r.success)

    print("\n" + "="*50)
    print("=============== Benchmark Summary ================")
    print("="*50)
    print(f"Total benchmark time: {duration:.2f} s")
    print(f"Total requests processed: {completed_requests} / {len(results)}")
    print(f"Throughput (requests/sec): {completed_requests / duration:.2f}")
    print(f"Throughput (output tokens/sec): {total_output_tokens / duration:.2f}")

    # --- Metrics Per Stage ---
    stage_metrics = {}
    for res in results:
        if res.stage not in stage_metrics:
            stage_metrics[res.stage] = []
        if res.success:
            stage_metrics[res.stage].append(res)
            
    print("\n" + "="*50)
    print("============ Stage-Specific Metrics =============")
    
    for stage_name, stage_results in stage_metrics.items():
        if not stage_results: continue
        
        # Use the constant stage duration for calculating throughput
        stage_duration = STAGE_DURATION_SECONDS 
        
        stage_completed = len(stage_results)
        stage_tokens = sum(r.output_len for r in stage_results)

        print(f"\n--- {stage_name} (Target RPS: {BENCHMARK_STAGES.get(stage_name)}) ---")
        print(f"  Requests Completed: {stage_completed}")
        print(f"  Stage Throughput (RPS): {stage_completed / stage_duration:.2f}")
        print(f"  Stage Throughput (Tokens/sec): {stage_tokens / stage_duration:.2f}")


    # --- Overall Latency Metrics ---

    ttfts, tpots, itls = [], [], []
    for res in results:
        if not res.success or res.output_len == 0: continue
        
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

    print("\n" + "-"*15 + "Time to First Token (Overall)" + "-"*15)
    print_latency_stats("TTFT", ttfts)
    print("\n" + "-----Time per Output Token (Overall, excl. 1st token)------")
    print_latency_stats("TPOT", tpots)
    print("\n" + "-"*15 + "Inter-token Latency (Overall)" + "-"*15)
    print_latency_stats("ITL", itls)
    print("="*50)


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A multi-stage, fixed-rate client using trace data content.")
    
    # Arguments for single target definition
    parser.add_argument("--host", type=str, default="localhost", help="Target host (e.g., NGINX IP).")
    parser.add_argument("--port", type=int, default=8888, help="Target port (e.g., NGINX proxy port).")
    parser.add_argument("--endpoint", type=str, default="/v1/completions", help="Target API endpoint.")
    
    # Argument for trace file
    parser.add_argument("--trace-file", type=str, required=True, help="Path to the JSONL trace file to sample request data from.")
    
    parser.add_argument("--model-name", type=str, default="NousResearch/Meta-Llama-3-8B-Instruct", help="The name of the model being served.")
    parser.add_argument(
        "--output-file",
        type=str,
        default="./staged_benchmark_results.jsonl",
        help="Path to a JSONL file to save detailed results for each request."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42, 
        help="Random seed for sampling to ensure reproducibility."
    )
    
    args = parser.parse_args()

    # 1. Form the single target URL
    api_url = f"http://{args.host}:{args.port}{args.endpoint}"

    # 2. Load all available requests from the trace file
    all_trace_requests = load_trace_file(args.trace_file)
    
    if not all_trace_requests:
        raise ValueError("No requests found in the trace file. Aborting benchmark.")

    # 3. Prepare and sample the requests for the entire benchmark run
    print(f"Loaded {len(all_trace_requests)} unique requests from trace file.")
    prepared_requests = prepare_requests(all_trace_requests, BENCHMARK_STAGES, STAGE_DURATION_SECONDS, args.seed)
    
    print(f"Benchmark will dispatch a total of {len(prepared_requests)} requests across all stages.")
    
    start_time = time.perf_counter()
    
    # 4. Run Benchmark
    results = asyncio.run(
        benchmark(
            api_url=api_url,
            model_name=args.model_name,
            stages=BENCHMARK_STAGES,
            stage_duration=STAGE_DURATION_SECONDS,
            prepared_requests=prepared_requests
        )
    )
    
    end_time = time.perf_counter()
    
    actual_duration = end_time - start_time
    
    # 5. Final Processing
    if args.output_file:
        save_results_to_file(results, args.output_file)
    
    calculate_metrics(results, actual_duration)