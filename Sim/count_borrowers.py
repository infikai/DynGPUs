import pandas as pd
import math
import argparse

# --- Constants from the simulator (components.py) ---
GPU_MEMORY_GB = 32
GPU_UTILIZATION_PERCENT = 100
SHARABLE_GPU_MEM_PENALTY_GB = 1.5

def count_borrow_eligible_jobs(file_path):
    """
    Loads a job workload from a CSV and counts how many training jobs
    are eligible to borrow GPUs for greedy speedup.
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
    effective_gpu_mem = GPU_MEMORY_GB - SHARABLE_GPU_MEM_PENALTY_GB

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

        # A job can borrow if it is NOT a high memory job
        if not is_high_memory_job:
            eligible_count += 1

    print("\n--- Analysis Complete ---")
    print(f"Total training jobs found: {len(training_df)}")
    print(f"Number of training jobs eligible to borrow GPUs: {eligible_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count borrow-eligible jobs in a simulator workload.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file containing the job workload.")
    args = parser.parse_args()
    count_borrow_eligible_jobs(args.csv_file)