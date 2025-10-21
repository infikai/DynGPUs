import pandas as pd
import math
import argparse

# --- Constants from the simulator (components.py) ---
GPU_MEMORY_GB = 32
GPU_UTILIZATION_PERCENT = 100
SHARABLE_GPU_MEM_PENALTY_GB = 1.5

def analyze_borrow_eligibility(file_path):
    """
    Loads a job workload from a CSV, counts eligible jobs, and lists
    the specific training jobs that are ineligible to borrow GPUs.
    """
    print(f"Analyzing workload from '{file_path}'...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    # --- Rename columns to match the simulator's internal names ---
    df.rename(columns={
        'start_time_t': 'arrival_time',
        'runtime': 'base_duration',
        'gpu_wrk_util': 'utilization_required',
        'max_gpu_wrk_mem': 'memory_required'
    }, inplace=True)

    # --- Filter for only training jobs ---
    training_df = df[df['task_group'].str.lower().str.contains('train', na=False)].copy()

    if training_df.empty:
        print("No training jobs found in the provided file.")
        return

    eligible_count = 0
    ineligible_jobs = [] # List to store details of ineligible jobs
    effective_gpu_mem = GPU_MEMORY_GB - SHARABLE_GPU_MEM_PENALTY_GB

    # Use .iterrows() to get the original DataFrame index for use as a Job ID
    for index, job in training_df.iterrows():
        # --- Calculate gpus_needed, mirroring the simulator's logic ---
        gpus_needed = max(
            math.ceil(job['memory_required'] / GPU_MEMORY_GB),
            math.ceil(job['utilization_required'] / GPU_UTILIZATION_PERCENT),
            1
        )

        if gpus_needed == 0:
            continue

        # --- Determine if the job is "high memory" ---
        mem_slice_per_gpu = job['memory_required'] / gpus_needed
        is_high_memory_job = mem_slice_per_gpu > effective_gpu_mem

        if not is_high_memory_job:
            eligible_count += 1
        else:
            # If the job is ineligible, store its details for later printing
            ineligible_jobs.append({
                'id': index,
                'memory_required': job['memory_required'],
                'gpus_needed': gpus_needed,
                'mem_per_gpu': mem_slice_per_gpu
            })

    print("\n--- Analysis Complete ---")
    print(f"Total training jobs found: {len(training_df)}")
    print(f"Number of training jobs eligible to borrow GPUs: {eligible_count}")

    # --- Print the detailed list of ineligible jobs ---
    if ineligible_jobs:
        print("\n--- Ineligible 'High Memory' Training Jobs ---")
        print(f"The following {len(ineligible_jobs)} jobs cannot borrow because their memory-per-GPU exceeds the {effective_gpu_mem:.2f} GB limit:")
        for job_info in ineligible_jobs:
            print(
                f"  - Job ID {job_info['id']}: "
                f"Requires {job_info['mem_per_gpu']:.2f} GB/GPU "
                f"(Total Mem: {job_info['memory_required']:.2f} GB / {job_info['gpus_needed']} GPUs)"
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze borrow-eligible jobs in a simulator workload.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file containing the job workload.")
    args = parser.parse_args()
    analyze_borrow_eligibility(args.csv_file)