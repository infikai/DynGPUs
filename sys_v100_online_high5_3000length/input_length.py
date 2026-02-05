import argparse
import json
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict

import numpy as np
# Requires: pip install transformers
from transformers import AutoTokenizer

# --- Global Configuration (Identical to previous) ---
STAGE_DURATION_SECONDS = 150  # 5 minutes per stage
BENCHMARK_STAGES = {
    "Stage 1": 0.2,
    "Stage 2": 0.3,
    "Stage 3": 0.5,
    "Stage 4": 0.5,
    "Stage 5": 0.3,
    "Stage 6": 0.5,
    "Stage 7": 0.3,
    "Stage 8": 0.5,
    "Stage 9": 0.3,
    "Stage 10": 0.5,
    "Stage 11": 0.7,
    "Stage 12": 0.5,
    "Stage 13": 0.3,
    "Stage 14": 0.2,
    "Stage 15": 0.4,
    "Stage 16": 0.5,
    "Stage 17": 0.2,
    "Stage 18": 0.3,
    "Stage 19": 0.5,
    "Stage 20": 0.5,
    "Stage 21": 0.2,
    "Stage 22": 0.3,
    "Stage 23": 0.5,
    "Stage 24": 0.2,
    "Stage 25": 0.5,
    "Stage 26": 0.2,
    "Stage 27": 0.3,
    "Stage 28": 0.4,
    "Stage 29": 0.5,
    "Stage 30": 0.2,
    "Stage 31": 0.3,
    "Stage 32": 0.5,
    "Stage 33": 0.4,
    "Stage 34": 0.2,
    "Stage 35": 0.5,
    "Stage 36": 0.2
}

@dataclass
class Request:
    timestamp: float
    prompt: str
    output_len: int
    request_id: int
    stage: str = "synthetic"

@dataclass
class TokenCountResult:
    request_id: int
    prompt_len: int
    target_output_len: int
    stage: str

class GlobalCounter:
    def __init__(self):
        self._count = 0
    def next(self):
        self._count += 1
        return self._count

# --- Original Logic Preservation ---

def load_trace_file(filepath: str) -> List[Request]:
    """Loads and parses the trace file identically to the original script."""
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
                print(f"Warning: Skipping malformed line {i+1}: {e}")
    return requests

def prepare_requests(all_requests: List[Request], stages: Dict[str, int], stage_duration: int, seed: int) -> List[Request]:
    """
    Samples requests identically to the original version.
    Uses the same seed and sampling logic to ensure consistency.
    """
    random.seed(seed)
    total_requests_needed = int(sum(rps * stage_duration for rps in stages.values()))
    
    if total_requests_needed == 0:
        return []

    if len(all_requests) < total_requests_needed:
        num_repeats = (total_requests_needed + len(all_requests) - 1) // len(all_requests)
        extended_requests = all_requests * num_repeats
    else:
        extended_requests = all_requests
        
    sampled_requests = random.sample(extended_requests, total_requests_needed)
    
    prepared_requests: List[Request] = []
    global_counter = GlobalCounter()
    current_index = 0
    
    for stage_name, rps in stages.items():
        requests_in_stage = int(rps * stage_duration)
        stage_requests = sampled_requests[current_index : current_index + requests_in_stage]
        
        for req in stage_requests:
            new_req = Request(
                timestamp=req.timestamp,
                prompt=req.prompt,
                output_len=req.output_len,
                request_id=global_counter.next(),
                stage=stage_name
            )
            prepared_requests.append(new_req)
        current_index += requests_in_stage
        
    return prepared_requests

# --- New Tokenization Logic ---

def process_locally(prepared_requests: List[Request], tokenizer_name: str) -> List[TokenCountResult]:
    """Iterates through the identically selected prompts and counts tokens."""
    print(f"Loading tokenizer: {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    results = []
    total = len(prepared_requests)
    
    for i, req in enumerate(prepared_requests):
        # Count tokens locally
        tokens = tokenizer.encode(req.prompt, add_special_tokens=True)
        
        results.append(TokenCountResult(
            request_id=req.request_id,
            prompt_len=len(tokens),
            target_output_len=req.output_len,
            stage=req.stage
        ))
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{total}...")
            
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Token Counter (Identical Sampling)")
    parser.add_argument("--trace-file", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-file", type=str, default="./token_stats.jsonl")
    
    args = parser.parse_args()

    # 1. Load Trace
    all_trace_requests = load_trace_file(args.trace_file)
    
    # 2. Prepare requests using the IDENTICAL logic and seed
    prepared_requests = prepare_requests(
        all_trace_requests, 
        BENCHMARK_STAGES, 
        STAGE_DURATION_SECONDS, 
        args.seed
    )
    
    # 3. Process locally instead of sending requests
    results = process_locally(prepared_requests, args.tokenizer)
    
    # 4. Save results
    with open(args.output_file, "w") as f:
        for res in results:
            f.write(json.dumps(asdict(res)) + "\n")
            
    print(f"Done. Processed {len(results)} prompts using seed {args.seed}.")