# file: run.py

import argparse
import pandas as pd
import time
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
    # Rename columns to match internal names if necessary
    df.rename(columns={
        'start_time_t': 'arrival_time',
        'runtime': 'base_duration',
        'gpu_wrk_util': 'utilization_required',
        'max_gpu_wrk_mem': 'memory_required'
    }, inplace=True)

    for index, row in df.iterrows():
        # Heuristic for job type
        if 'train' in str(row.get('task_group', '')).lower():
            job_type = 'training'
        else:
            job_type = 'inference'

        job = Job(id=f"job_{index}",
                  job_type=job_type,
                  base_duration=row['base_duration'],
                  arrival_time=row['arrival_time'],
                  memory_required=row['memory_required'],
                  utilization_required=row['utilization_required'])
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
    # LLM Constants (mirrored from components for estimation if needed)
    LLM_BASE_TTFT = 2.5
    LLM_TKN_PER_INPUT = 0.005
    LLM_TPOT = 0.1

    df.rename(columns={
        'TIMESTAMP_seconds': 'arrival_time',
        'ContextTokens': 'input_tokens',
        'GeneratedTokens': 'output_tokens'
    }, inplace=True)

    for row in df.itertuples():
        # Duration is calculated inside Job.__init__ if tokens are passed,
        # but we pass tokens here explicitly.
        jobs.append(Job(id=f"llm_{row.Index}",
                        job_type='llm_inference',
                        arrival_time=getattr(row, 'arrival_time', 0),
                        input_tokens=getattr(row, 'input_tokens', 0),
                        output_tokens=getattr(row, 'output_tokens', 0)))
    return jobs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepBoot Simulator")
    parser.add_argument("csv_file", type=str, help="Path to training workload CSV")
    parser.add_argument("--llm-trace", type=str, help="Path to LLM inference trace CSV")
    
    # New arguments required by the updated Scheduler
    parser.add_argument("--progress-interval", type=int, default=1000, help="Ticks between progress prints")
    parser.add_argument("--log-interval", type=int, default=100, help="Ticks between usage logs")
    parser.add_argument("--start-time", type=int, default=0, help="Simulation start timestamp")
    parser.add_argument("--end-time", type=int, default=-1, help="Simulation end timestamp (-1 for auto)")
    parser.add_argument("--tick-duration", type=int, default=1, help="Seconds per simulation tick")

    args = parser.parse_args()

    # 1. Initialize Cluster (DeepBoot Split: 1000 Training, 1000 Inference)
    cluster = ClusterManager(num_training_gpus=1000, num_inference_gpus=1000)

    # 2. Load Jobs
    workload = load_jobs_from_csv(args.csv_file)
    if args.llm_trace:
        workload.extend(load_llm_jobs_from_csv(args.llm_trace))
    
    # Sort entire workload by arrival time
    workload.sort(key=lambda j: j.arrival_time)

    # 3. Start Scheduler with ALL required arguments
    if workload:
        scheduler = Scheduler(
            jobs_list=workload, 
            cluster_manager=cluster,
            progress_interval=args.progress_interval,
            log_interval=args.log_interval,
            start_time=args.start_time,
            end_time=args.end_time,
            tick_duration=args.tick_duration
        )
        scheduler.run_simulation()
    else:
        print("No jobs found to simulate.")