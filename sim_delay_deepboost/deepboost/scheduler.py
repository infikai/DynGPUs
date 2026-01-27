# file: scheduler.py

import math
from collections import deque
# --- (MODIFIED: Removed preemption/reclamation constants) ---
from components import (SimulationClock, Job, GPU, GPU_MEMORY_GB, GPU_UTILIZATION_PERCENT, 
                        LLM_MAX_CONCURRENCY)

class Scheduler:
    """
    Manages job scheduling for a heterogeneous GPU cluster.
    """
    def __init__(self, jobs_list, cluster_manager, progress_interval, log_interval, start_time, end_time, tick_duration):
        self.pending_jobs = deque(sorted(jobs_list, key=lambda j: j.arrival_time))
        self.running_jobs = []
        self.completed_jobs = []
        self.cluster = cluster_manager
        self.clock = SimulationClock(tick_duration=tick_duration)
        # --- (REMOVED preemption/reclamation attributes) ---
        # self.preemption_count = 0
        # self.reclamation_count = 0
        # self.preemption_map = {}

        # Intervals controlled by user input
        self.progress_interval = progress_interval
        self.log_interval = log_interval
        self.start_time = start_time
        self.end_time = end_time

        # --- (NEW: Attributes for interval-based delay logging) ---
        self.delay_log_interval = 600 # 10 minutes
        self.next_delay_log_time = 0 # Will be set in run_simulation
        self.current_inference_delays = [] # Stores delays for jobs completed in the interval

        # Initialize log files
        self.training_log_file = open("training_job_log_hvd.csv", "w")
        # --- (MODIFIED: Changed from inference_job_log to inference_delay_log) ---
        self.inference_delay_log_file = open("inference_delay_log_hvd.csv", "w")
        self.usage_log_file = open("gpu_usage_log_hvd.csv", "w")
        self._initialize_logs()

    def _initialize_logs(self):
        """Writes headers to the log files."""
        # --- (MODIFIED: Added start_time and delay) ---
        self.training_log_file.write("job_id,arrival_time,start_time,delay,base_duration,ideal_completion_time,actual_completion_time,performance_factor,gpus\n")
        # --- (MODIFIED: Header for interval-based delay log) ---
        self.inference_delay_log_file.write("timestamp,average_delay_seconds,job_count\n")
        self.usage_log_file.write("timestamp,training_gpus_used,inference_gpus_used,borrowed_inference_gpus\n")

    def _log_gpu_usage(self):
        """
        Logs a snapshot of GPU usage by directly inspecting the state of each GPU.
        """
        training_gpus_used = 0
        inference_gpus_used = 0
        borrowed_gpus_used = 0 # This will now always be 0, but we keep it for log consistency

        # 1. Count usage in the dedicated training pool
        for gpu in self.cluster.training_gpus:
            if not gpu.is_idle():
                training_gpus_used += 1

        # 2. Count usage in the inference pool with the new logic
        for gpu in self.cluster.inference_gpus:
            
            if gpu.is_llm_server:
                inference_gpus_used += 1
                
            elif not gpu.is_idle():
                has_native_task = False
                # has_borrowed_task = False # No longer possible

                for task in gpu.running_tasks.values():
                    job_type = task['job'].job_type
                    
                    if job_type == 'inference': # No 'llm_inference' needed here
                        has_native_task = True
                    # (No training jobs can run here anymore)

                if has_native_task:
                    inference_gpus_used += 1
        
        self.usage_log_file.write(f"{self.clock.current_time},{training_gpus_used},{inference_gpus_used},{borrowed_gpus_used}\n")
    
    def _dispatch_job(self, job):
        """Routes a job to the appropriate dispatch method based on its type."""
        if job.job_type == 'training':
            return self._dispatch_training_job(job)
        elif job.job_type == 'inference':
            return self._dispatch_inference_job(job)
        elif job.job_type == 'llm_inference':
            return self._dispatch_llm_inference_job(job)
        return False
    
    def _batch_dispatch_llm_jobs(self, llm_jobs):
        """
        Dispatches a batch of LLM jobs by intelligently filling available GPUs.
        This function now explicitly handles GPU conversion.
        """
        if not llm_jobs:
            return []

        num_jobs_to_assign = len(llm_jobs)
        # Get the list of *GPUs* we can use, sorted by priority
        available_gpus, _ = self.cluster.find_resources_for_llm_batch(num_jobs_to_assign)
        
        assigned_count = 0
        job_index = 0
        
        # Iterate through the available GPUs one by one
        for gpu in available_gpus:
            if job_index >= num_jobs_to_assign:
                break # All jobs have been assigned

            slots_to_fill = 0
            
            # Explicitly check if the GPU needs conversion
            if not gpu.is_llm_server:
                # This is an idle GPU. Convert it now.
                was_converted = gpu.convert_to_llm_server()
                if not was_converted:
                    continue # Conversion failed (maybe not idle?), skip this GPU
                
                slots_to_fill = gpu.llm_slots_available
            else:
                slots_to_fill = gpu.llm_slots_available
                
            # Fill this one GPU with as many jobs as it can take
            for _ in range(slots_to_fill):
                if job_index >= num_jobs_to_assign:
                    break 
                
                job = llm_jobs[job_index]
                job.start_time = self.clock.current_time
                job.assigned_gpus = [gpu]
                
                # --- (NEW: Log delay on job start) ---
                delay = math.floor(max(0, job.start_time - job.arrival_time))
                if delay > 0:
                    self.current_inference_delays.append(delay)
                
                gpu.assign_llm_task(job) 
                self.running_jobs.append(job)
                
                assigned_count += 1
                job_index += 1

        # Return any jobs that are left over
        unassigned_jobs = llm_jobs[assigned_count:]
        return unassigned_jobs

    def _dispatch_llm_inference_job(self, job):
        """
        Schedules a single LLM inference request.
        """
        # 1. Find a GPU that is either an
        #    active LLM server with slots or an idle GPU ready for conversion.
        gpu = self.cluster.find_gpu_for_llm_job()
        
        # 2. If no GPU is readily available, the job must wait.
        #    (Preemption logic removed)
        
        # 3. If we have a GPU, assign the job.
        if gpu:
            job.assigned_gpus = [gpu]
            job.start_time = self.clock.current_time
            
            # --- (NEW: Log delay on job start) ---
            delay = max(0, job.start_time - job.arrival_time)
            self.current_inference_delays.append(delay)
            
            # --- (MODIFIED: Explicitly convert if not already a server) ---
            if not gpu.is_llm_server:
                gpu.convert_to_llm_server()
            
            gpu.assign_llm_task(job)
            self.running_jobs.append(job)
            return True
            
        # 4. If no GPU could be found, the job must wait.
        return False

    def _dispatch_inference_job(self, job):
        """Routes an inference job to the correct scheduler based on its size."""
        is_large_job = (job.memory_required > GPU_MEMORY_GB or 
                        job.utilization_required > GPU_UTILIZATION_PERCENT)
        
        if is_large_job:
            return self._dispatch_large_inference_job(job)
        else:
            return self._dispatch_stackable_inference_job(job)

    def _dispatch_stackable_inference_job(self, job):
        """Schedules a small inference job."""
        job.gpus_needed = 1 
        gpu = self.cluster.find_gpu_for_stackable_inference(job)
        if gpu:
            job.assign_resources([gpu], self.clock.current_time)
            
            # --- (NEW: Log delay on job start) ---
            delay = max(0, job.start_time - job.arrival_time)
            self.current_inference_delays.append(delay)
            
            self.running_jobs.append(job)
            return True

        # If no space, job must wait. (Preemption logic removed)
        return False

    def _dispatch_large_inference_job(self, job):
        """Schedules a large inference job."""
        effective_gpu_mem = GPU_MEMORY_GB
        gpus_needed = max(math.ceil(job.memory_required / effective_gpu_mem),
                          math.ceil(job.utilization_required / GPU_UTILIZATION_PERCENT), 1)
        job.gpus_needed = gpus_needed
        
        allocated_gpus = self.cluster.find_idle_gpus_in_inference_pool(gpus_needed)
        
        # If not enough idle GPUs, job must wait. (Preemption logic removed)
        if len(allocated_gpus) == gpus_needed:
            job.assign_resources(allocated_gpus, self.clock.current_time)
            
            # --- (NEW: Log delay on job start) ---
            delay = max(0, job.start_time - job.arrival_time)
            self.current_inference_delays.append(delay)
            
            self.running_jobs.append(job)
            return True
            
        return False
        
    def _dispatch_training_job(self, job):
        """
        Schedules a training job. 
        It can *only* use GPUs from the dedicated training pool.
        If not enough GPUs are available, the job waits.
        """
        gpus_needed = max(math.ceil(job.memory_required / GPU_MEMORY_GB),
                          math.ceil(job.utilization_required / GPU_UTILIZATION_PERCENT), 1)
        job.gpus_needed = gpus_needed

        allocated_gpus = self.cluster.find_idle_gpus_for_training(gpus_needed)
        
        if len(allocated_gpus) == gpus_needed:
            job.assign_resources(allocated_gpus, self.clock.current_time)
            self.running_jobs.append(job)
            return True
            
        return False
         
    def _handle_job_completion(self, job):
        """Processes a finished job, logs training data, and handles LLM scale-down."""
        freed_gpus = list(job.assigned_gpus)
        self.cluster.release_resources_for_job(job) # This calls gpu.release_task()
        job.record_completion(self.clock.current_time)
        self.running_jobs.remove(job)
        self.completed_jobs.append(job)

        # --- (MODIFIED: Calculate delay for all jobs) ---
        delay = 0
        if job.start_time > job.arrival_time:
             delay = job.start_time - job.arrival_time

        if job.job_type == 'training':
            ideal_completion_time = job.arrival_time + job.base_duration
            actual_duration = job.completion_time - job.arrival_time
            perf_factor = actual_duration / job.base_duration if job.base_duration > 0 else 0
            
            # --- (MODIFIED: Added start_time and delay to log entry) ---
            log_entry = (f"{job.id},{job.arrival_time},{job.start_time},{delay:.2f},{job.base_duration},"
                         f"{ideal_completion_time},{job.completion_time},{perf_factor:.4f},{job.gpus_needed}\n")
            self.training_log_file.write(log_entry)

        # --- (MODIFIED: Removed delay logging from here) ---
        # elif job.job_type in ['inference', 'llm_inference']:
        #     self.current_inference_delays.append(delay)


        # --- (MODIFIED: Simplified to only handle LLM scale-down) ---
        if job.job_type == 'llm_inference':
            for gpu in freed_gpus:
                # On-demand scale-down
                if gpu.is_llm_server and not gpu.running_tasks:
                    print(f"💡 Clock {self.clock.current_time}: Reverting empty LLM server {gpu.gpu_id} to regular GPU.")
                    gpu.revert_from_llm_server()
                
                # (Reclamation logic removed)
    
    # --- (NEW: Method to log average inference delay) ---
    def _log_average_inference_delay(self):
        """Calculates and logs the average inference delay for the completed interval."""
        if not self.current_inference_delays:
            # Log 0 if no inference jobs completed in this interval
            avg_delay = 0
            job_count = 0
        else:
            avg_delay = sum(self.current_inference_delays) / len(self.current_inference_delays)
            job_count = len(self.current_inference_delays)

        # Log at the *end* of the interval, which is the current time
        log_entry = f"{self.clock.current_time},{avg_delay:.2f},{job_count}\n"
        self.inference_delay_log_file.write(log_entry)
        
        # Reset the list for the next interval
        self.current_inference_delays = []

    def run_simulation(self):
        """Main simulation loop with adaptive policies and fair dispatching."""
        if not self.pending_jobs: 
            print("No jobs to simulate.")
            self.print_results()
            return

        # --- Initialization and Filtering ---
        effective_start_time = self.start_time
        if self.start_time == 0:
            effective_start_time = self.pending_jobs[0].arrival_time
            print(f"No specific start time given. Fast-forwarding to first job arrival at time {effective_start_time}.")

        original_job_count = len(self.pending_jobs)
        filtered_list = [j for j in self.pending_jobs if j.arrival_time >= effective_start_time]
        self.pending_jobs = deque(filtered_list)
        print(f"Filtered out {original_job_count - len(self.pending_jobs)} jobs that arrived before effective start time {effective_start_time}.")
        
        if not self.pending_jobs: 
            print("No jobs to simulate in the specified time window.")
            self.print_results()
            return

        self.clock.current_time = effective_start_time
        # --- (NEW: Set the first delay log time aligned to the interval) ---
        self.next_delay_log_time = ( (effective_start_time // self.delay_log_interval) + 1 ) * self.delay_log_interval
        
        self.jobs_to_retry = deque() # Initialize the retry queue

        # --- Main Simulation Loop ---
        while self.pending_jobs or self.running_jobs or self.jobs_to_retry:
            if self.end_time != -1 and self.clock.current_time >= self.end_time:
                print(f"\n🛑 Simulation ended at specified end time: {self.end_time}")
                break
            
            self.clock.tick()

            # --- (NEW) Periodic Inference Delay Logging ---
            if self.clock.current_time >= self.next_delay_log_time:
                self._log_average_inference_delay()
                self.next_delay_log_time += self.delay_log_interval

            # --- Periodic Logging Calls (at the start of the tick) ---
            if self.clock.current_time > 0:
                # (Policy call removed)
                
                # Log GPU usage
                if self.clock.current_time % self.log_interval == 0:
                    self._log_gpu_usage()
            
            if self.clock.current_time % self.progress_interval == 0 and (self.running_jobs or self.pending_jobs or self.jobs_to_retry):
                
                num_llm_servers = sum(1 for gpu in self.cluster.inference_gpus if gpu.is_llm_server)
                num_llm_jobs = sum(1 for job in self.running_jobs if job.job_type == 'llm_inference')
                
                print(f"🕒 Clock {self.clock.current_time}: "
                      f"Pending={len(self.pending_jobs)}, "
                      f"Retrying={len(self.jobs_to_retry)}, "
                      f"Running={len(self.running_jobs)} (LLM: {num_llm_jobs}), "
                      f"LLM Servers={num_llm_servers}, "
                      f"Completed={len(self.completed_jobs)}")
            
            # --- Fairer Dispatch Logic (prevents Head-of-Line Blocking) ---
            arrived_jobs = list(self.jobs_to_retry)
            self.jobs_to_retry.clear()
            while self.pending_jobs and self.pending_jobs[0].arrival_time <= self.clock.current_time:
                arrived_jobs.append(self.pending_jobs.popleft())

            if arrived_jobs:
                arrived_llm_jobs = [j for j in arrived_jobs if j.job_type == 'llm_inference']
                other_arrived_jobs = [j for j in arrived_jobs if j.job_type != 'llm_inference']
                
                unassigned_llm_jobs = self._batch_dispatch_llm_jobs(arrived_llm_jobs)

                unassigned_other_jobs = []
                for job in other_arrived_jobs:
                    if not self._dispatch_job(job):
                        unassigned_other_jobs.append(job)

                all_unassigned = sorted(unassigned_llm_jobs + unassigned_other_jobs, key=lambda j: j.arrival_time)
                self.jobs_to_retry.extend(all_unassigned)
            
            # --- Job Progress and Completion Handling ---
            finished_this_tick = []
            for job in self.running_jobs:
                job.update_progress(self.clock.tick_duration, self.clock.current_time)
                if job.is_complete():
                    finished_this_tick.append(job)
            
            for job in finished_this_tick:
                self._handle_job_completion(job)


    def print_results(self):
        """Prints a final summary and saves it to simulation_summary.txt."""
        
        # --- (NEW: Log any remaining inference delays before closing) ---
        print(f"Logging remaining inference delays at final time {self.clock.current_time}...")
        self._log_average_inference_delay()

        self.training_log_file.close()
        # --- (MODIFIED: Close inference_delay_log_file) ---
        self.inference_delay_log_file.close()
        self.usage_log_file.close()

        with open("simulation_summary.txt", "w") as summary_file:
            total_jobs = len(self.completed_jobs)
            if total_jobs == 0:
                summary_text = "No jobs were completed in the simulation window."
                print(summary_text)
                summary_file.write(summary_text + "\n")
                return

            training_jobs = [j for j in self.completed_jobs if j.job_type == 'training']
            inference_jobs = [j for j in self.completed_jobs if j.job_type == 'inference']
            llm_inference_jobs = [j for j in self.completed_jobs if j.job_type == 'llm_inference']
            
            # Combine regular and LLM inference for the average
            all_inference_jobs = inference_jobs + llm_inference_jobs
            
            avg_training_turnaround = sum(j.turnaround_time for j in training_jobs) / len(training_jobs) if training_jobs else 0
            avg_inference_turnaround = sum(j.turnaround_time for j in all_inference_jobs) / len(all_inference_jobs) if all_inference_jobs else 0
            
            # --- (NEW: Calculate average delays) ---
            avg_training_delay = sum(j.start_time - j.arrival_time for j in training_jobs if j.start_time != -1) / len(training_jobs) if training_jobs else 0
            avg_inference_delay = sum(j.start_time - j.arrival_time for j in all_inference_jobs if j.start_time != -1) / len(all_inference_jobs) if all_inference_jobs else 0

            # Build the output lines
            lines = [
                "--- Simulation Results ---",
                # --- (MODIFIED: Updated log file name in message) ---
                f"Detailed logs saved to 'training_job_log.csv', 'inference_delay_log.csv', and 'gpu_usage_log.csv'",
                f"Total Jobs Completed: {total_jobs}",
                # --- (REMOVED preemption/reclamation lines) ---
                # f"Total Preemptions: {self.preemption_count}",
                # f"Total Successful Reclamations: {self.reclamation_count}",
                f"Average Training Job Turnaround: {avg_training_turnaround:.2f} seconds",
                f"Average Inference Job Turnaround: {avg_inference_turnaround:.2f} seconds",
                # --- (NEW: Report average delays) ---
                f"Average Training Job Delay (Queue Time): {avg_training_delay:.2f} seconds",
                f"Average Inference Job Delay (Queue Time): {avg_inference_delay:.2f} seconds",
                "--------------------------"
            ]

            # Print to console and write to the summary file
            for line in lines:
                print(line)
                summary_file.write(line + "\n")