# file: run.py

import argparse
import pandas as pd
from cluster_manager import ClusterManager
from scheduler import Scheduler
from components import Job

def load_jobs_from_csv(file_path):
    print(f"Loading jobs from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []

    jobs = []
    for index, row in df.iterrows():
        j_type = 'training' if 'train' in str(row.get('task_group', '')).lower() else 'inference'
        job = Job(id=f"job_{index}",
                  job_type=j_type,
                  base_duration=row['runtime'] if 'runtime' in row else row.get('base_duration', 10),
                  arrival_time=row['start_time_t'] if 'start_time_t' in row else row.get('arrival_time', 0),
                  memory_required=row['max_gpu_wrk_mem'] if 'max_gpu_wrk_mem' in row else 1)
        jobs.append(job)
    return jobs

def load_llm_jobs_from_csv(file_path):
    print(f"Loading LLM jobs from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading LLM trace: {e}")
        return []

    jobs = []
    LLM_BASE_TTFT = 2.5
    LLM_TKN_PER_INPUT = 0.005
    LLM_TPOT = 0.1

    for row in df.itertuples():
        input_tokens = getattr(row, 'ContextTokens', 0)
        output_tokens = getattr(row, 'GeneratedTokens', 0)
        duration = LLM_BASE_TTFT + (LLM_TKN_PER_INPUT * input_tokens) + (LLM_TPOT * output_tokens)
        
        jobs.append(Job(id=f"llm_{row.Index}",
                        job_type='llm_inference',
                        arrival_time=getattr(row, 'TIMESTAMP_seconds', 0),
                        base_duration=duration))
    return jobs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepBoot Simulator")
    parser.add_argument("csv_file", type=str, help="Path to training workload CSV")
    parser.add_argument("--llm-trace", type=str, help="Path to LLM inference trace CSV")
    args = parser.parse_args()

    # 1. Initialize Cluster (DeepBoot Split: 1000 Training, 1000 Inference)
    cluster = ClusterManager(num_training_gpus=485, num_inference_gpus=615)

    # 2. Load Jobs
    workload = load_jobs_from_csv(args.csv_file)
    if args.llm_trace:
        workload.extend(load_llm_jobs_from_csv(args.llm_trace))
    
    # Sort entire workload by arrival time
    workload.sort(key=lambda j: j.arrival_time)

    # 3. Start Scheduler
    if workload:
        scheduler = Scheduler(workload, cluster, tick_duration=1)
        scheduler.run_simulation()
    else:
        print("No jobs found to simulate.")