# file: run_simulation.py

import pandas as pd
import argparse
from scheduler import Scheduler
from cluster_manager import ClusterManager
from components import Job

def load_jobs_from_csv(file_path):
    """
    Loads job definitions from a specified CSV file and counts job types.
    """
    print(f"Loading jobs from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []

    jobs = []
    # ** NEW: Initialize counters **
    train_count = 0
    inference_count = 0

    df.rename(columns={
        'start_time_t': 'arrival_time',
        'runtime': 'base_duration',
        'gpu_wrk_util': 'utilization_required',
        'max_gpu_wrk_mem': 'memory_required'
    }, inplace=True)

    for index, row in df.iterrows():
        # Heuristic to determine job type
        if 'train' in str(row['task_group']).lower():
            job_type = 'training'
            train_count += 1 # Increment training counter
        else:
            job_type = 'inference'
            inference_count += 1 # Increment inference counter

        job = Job(id=f"job_{index}",
                  job_type=job_type,
                  base_duration=row['base_duration'],
                  arrival_time=row['arrival_time'],
                  memory_required=row['memory_required'],
                  utilization_required=row['utilization_required'])
        jobs.append(job)
        
    print(f"Successfully loaded {len(jobs)} total jobs.")
    # ** NEW: Print the counts **
    print(f"➡️ Found {train_count} training jobs and {inference_count} inference jobs in the workload.")
    return jobs

def load_llm_jobs_from_csv(file_path):
    """
    Loads LLM inference job definitions from a specified CSV file.
    """
    print(f"Loading LLM jobs from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Warning: The file '{file_path}' was not found. No LLM jobs will be loaded.")
        return []

    jobs = []
    # Assumes columns are named: arrival_time, input_tokens, output_tokens
    df.rename(columns={
        'start_time_t': 'arrival_time', # Accommodate different column names
        'in_tokens': 'input_tokens',
        'out_tokens': 'output_tokens'
    }, inplace=True)

    for index, row in df.iterrows():
        job = Job(id=f"llm_job_{index}",
                  job_type='llm_inference',
                  arrival_time=row['arrival_time'],
                  input_tokens=row['input_tokens'],
                  output_tokens=row['output_tokens'])
        jobs.append(job)
        
    print(f"➡️ Successfully loaded {len(jobs)} LLM inference jobs.")
    return jobs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced GPU Cluster Simulator")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file containing the job workload.")
    parser.add_argument("--llm-trace", type=str, help="Path to the CSV file with LLM inference requests.")
    parser.add_argument("--progress-interval", type=int, default=1000, help="Interval for printing progress to console.")
    parser.add_argument("--log-interval", type=int, default=1000, help="Interval for logging GPU usage to file.")
    parser.add_argument("--policy-interval", type=int, default=500, help="Interval for checking the dynamic locking policy.")
    parser.add_argument("--start-time", type=int, default=0, help="Simulation start time.")
    parser.add_argument("--end-time", type=int, default=-1, help="Simulation end time.")
    # ** NEW: Add optional argument for tick_duration **
    parser.add_argument("--tick-duration", type=int, default=1, help="The duration of each simulation time step (tick).")
    args = parser.parse_args()

    print("🚀 Starting Simulation...")
    cluster = ClusterManager(num_training_gpus=1700, 
                             num_inference_gpus=170)
    
    job_workload = load_jobs_from_csv(args.csv_file)
    if args.llm_trace:
        llm_job_workload = load_llm_jobs_from_csv(args.llm_trace)
        job_workload.extend(llm_job_workload)
    
    # CRITICAL: Re-sort the combined list by arrival time
    job_workload.sort(key=lambda j: j.arrival_time)
    
    if job_workload:
        # ** MODIFIED: Pass tick_duration to the Scheduler **
        scheduler = Scheduler(job_workload, cluster, 
                              progress_interval=args.progress_interval,
                              log_interval=args.log_interval,
                              policy_interval=args.policy_interval,
                              start_time=args.start_time,
                              end_time=args.end_time,
                              tick_duration=args.tick_duration)
        scheduler.run_simulation()
        
        print("\n✅ Simulation Finished.")
        scheduler.print_results()
    else:
        print("Simulation aborted due to missing workload.")
