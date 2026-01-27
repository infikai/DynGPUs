# file: scheduler.py

import math
from collections import deque
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

        # Intervals controlled by user input
        self.progress_interval = progress_interval
        self.log_interval = log_interval
        self.start_time = start_time
        self.end_time = end_time

        # Attributes for interval-based delay logging
        self.delay_log_interval = 600 # 10 minutes
        self.next_delay_log_time = 0 # Will be set in run_simulation
        self.current_inference_delays = [] # Stores delays for jobs completed in the interval

        # Initialize log files
        self.training_log_file = open("training_job_log_hvd.csv", "w")
        self.inference_delay_log_file = open("inference_delay_log_hvd.csv", "w")
        self.usage_log_file = open("gpu_usage_log_hvd.csv", "w")
        self._initialize_logs()

    def _initialize_logs(self):
        """Writes headers to the log files."""
        self.training_log_file.write("job_id,arrival_time,start_time,delay,base_duration,ideal_completion_time,actual_completion_time,performance_factor,gpus\n")
        self.inference_delay_log_file.write("timestamp,average_delay_seconds,job_count\n")
        self.usage_log_file.write("timestamp,training_gpus_used,inference_gpus_used,borrowed_inference_gpus\n")

    def _log_gpu_usage(self):
        """
        Logs a snapshot of GPU usage by directly inspecting the state of each GPU.
        """
        training_gpus_used = 0
        inference_gpus_used = 0
        borrowed_gpus_used = 0 

        # 1. Count usage in the dedicated training pool
        for gpu in self.cluster.training_gpus:
            if not gpu.is_idle():
                training_gpus_used += 1

        # 2. Count usage in the inference pool
        for gpu in self.cluster.inference_gpus:
            if gpu.is_llm_server:
                inference_gpus_used += 1
            elif not gpu.is_idle():
                inference_gpus_used += 1
                
                # Check if it is actually a borrowed job (training job on inference GPU)
                for task in gpu.running_tasks.values():
                    if task['job'].job_type == 'training':
                         borrowed_gpus_used += 1
                         # Correct the counters: move from inference_used to borrowed
                         inference_gpus_used -= 1
                         break
        
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
        """Dispatches a batch of LLM jobs by intelligently filling available GPUs."""
        if not llm_jobs:
            return []

        num_jobs_to_assign = len(llm_jobs)
        available_gpus, _ = self.cluster.find_resources_for_llm_batch(num_jobs_to_assign)
        
        assigned_count = 0
        job_index = 0
        
        for gpu in available_gpus:
            if job_index >= num_jobs_to_assign:
                break 

            slots_to_fill = 0
            if not gpu.is_llm_server:
                was_converted = gpu.convert_to_llm_server()
                if not was_converted:
                    continue 
                slots_to_fill = gpu.llm_slots_available
            else:
                slots_to_fill = gpu.llm_slots_available
                
            for _ in range(slots_to_fill):
                if job_index >= num_jobs_to_assign:
                    break 
                
                job = llm_jobs[job_index]
                job.start_time = self.clock.current_time
                job.assigned_gpus = [gpu]
                
                delay = math.floor(max(0, job.start_time - job.arrival_time))
                if delay > 0:
                    self.current_inference_delays.append(delay)
                
                gpu.assign_llm_task(job) 
                self.running_jobs.append(job)
                
                assigned_count += 1
                job_index += 1

        unassigned_jobs = llm_jobs[assigned_count:]
        return unassigned_jobs

    def _dispatch_llm_inference_job(self, job):
        """Schedules a single LLM inference request."""
        gpu = self.cluster.find_gpu_for_llm_job()
        
        if gpu:
            job.assigned_gpus = [gpu]
            job.start_time = self.clock.current_time
            
            delay = max(0, job.start_time - job.arrival_time)
            self.current_inference_delays.append(delay)
            
            if not gpu.is_llm_server:
                gpu.convert_to_llm_server()
            
            gpu.assign_llm_task(job)
            self.running_jobs.append(job)
            return True
            
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
            delay = max(0, job.start_time - job.arrival_time)
            self.current_inference_delays.append(delay)
            self.running_jobs.append(job)
            return True
        return False

    def _dispatch_large_inference_job(self, job):
        """Schedules a large inference job."""
        effective_gpu_mem = GPU_MEMORY_GB
        gpus_needed = max(math.ceil(job.memory_required / effective_gpu_mem),
                          math.ceil(job.utilization_required / GPU_UTILIZATION_PERCENT), 1)
        job.gpus_needed = gpus_needed
        
        allocated_gpus = self.cluster.find_idle_gpus_in_inference_pool(gpus_needed)
        
        if len(allocated_gpus) == gpus_needed:
            job.assign_resources(allocated_gpus, self.clock.current_time)
            delay = max(0, job.start_time - job.arrival_time)
            self.current_inference_delays.append(delay)
            self.running_jobs.append(job)
            return True
        return False
        
    def _dispatch_training_job(self, job):
        """
        Schedules a training job using a Greedy approach with Borrowing.
        The job can start with fewer than desired GPUs if at least 1 is available.
        """
        # 1. Calculate ideal resources needed
        desired_gpus = max(math.ceil(job.memory_required / GPU_MEMORY_GB),
                           math.ceil(job.utilization_required / GPU_UTILIZATION_PERCENT), 1)
        # We clamp specific logic: usually training wants 4, but let's stick to calculation or 4
        if desired_gpus < 4 and job.memory_required > 1: # Heuristic: if it's a "real" training job
            desired_gpus = 4
        
        job.gpus_needed = desired_gpus
        min_gpus = 1 # Can start with 1

        assigned_gpus = []

        # 2. Priority 1: Take as many idle dedicated Training GPUs as possible
        for gpu in self.cluster.training_gpus:
            if gpu.is_idle():
                assigned_gpus.append(gpu)
                if len(assigned_gpus) == desired_gpus:
                    break
        
        # 3. Priority 2: Borrow from Inference Pool if still needed
        if len(assigned_gpus) < desired_gpus:
            needed = desired_gpus - len(assigned_gpus)
            for gpu in self.cluster.inference_gpus:
                if gpu.is_idle():
                    assigned_gpus.append(gpu)
                    if len(assigned_gpus) == desired_gpus:
                        break
        
        # 4. Dispatch if we met the MINIMUM requirement (elastic start)
        if len(assigned_gpus) >= min_gpus:
            # Mark borrowed GPUs
            for gpu in assigned_gpus:
                if gpu.gpu_type == 'inference':
                    gpu.state = 'TRAIN'
                    gpu.protect_time_remaining = 0

            job.assign_resources(assigned_gpus, self.clock.current_time)
            self.running_jobs.append(job)
            return True
            
        return False

    def _try_scale_up_training_jobs(self):
        """
        Iterates through running training jobs that have fewer GPUs than desired
        and tries to acquire newly available resources (Elastic Scaling).
        """
        for job in self.running_jobs:
            if job.job_type == 'training':
                current_count = len(job.assigned_gpus)
                # Desired is stored in gpus_needed, or default to 4 if not set correctly
                desired = getattr(job, 'gpus_needed', 4) 
                
                if current_count < desired:
                    needed = desired - current_count
                    newly_acquired = []

                    # 1. Try Dedicated Training Pool first
                    for gpu in self.cluster.training_gpus:
                        if gpu.is_idle():
                            newly_acquired.append(gpu)
                            if len(newly_acquired) == needed:
                                break
                    
                    # 2. Try Borrowing from Inference Pool
                    if len(newly_acquired) < needed:
                        remaining_need = needed - len(newly_acquired)
                        for gpu in self.cluster.inference_gpus:
                            if gpu.is_idle():
                                newly_acquired.append(gpu)
                                if len(newly_acquired) >= remaining_need: # Correction: append to accumulated list
                                    break
                        # Fix logic: we have 'newly_acquired' mixing both pools. 
                        # We stop if the total found equals needed.
                        if len(newly_acquired) > needed:
                            newly_acquired = newly_acquired[:needed]

                    # 3. Apply Scaling
                    if newly_acquired:
                        # We must re-assign the job to re-distribute memory/load.
                        # Since assign_resources clears current usage, we must:
                        # a. Release current
                        old_gpus = list(job.assigned_gpus)
                        for gpu in old_gpus:
                            gpu.release_task(job)
                        
                        # b. Combine lists
                        all_gpus = old_gpus + newly_acquired
                        
                        # c. Set state for new GPUs
                        for gpu in newly_acquired:
                             if gpu.gpu_type == 'inference':
                                gpu.state = 'TRAIN'
                                gpu.protect_time_remaining = 0
                        
                        # d. Re-assign with ORIGINAL start time (preserve queue metrics)
                        job.assign_resources(all_gpus, job.start_time)
                        
                        # Optional: Log the scaling event
                        # print(f"📈 Clock {self.clock.current_time}: Job {job.id} scaled up from {current_count} to {len(all_gpus)} GPUs.")

    def _handle_job_completion(self, job):
        """Processes a finished job, logs training data, and handles LLM scale-down."""
        freed_gpus = list(job.assigned_gpus)
        self.cluster.release_resources_for_job(job) 
        job.record_completion(self.clock.current_time)
        self.running_jobs.remove(job)
        self.completed_jobs.append(job)

        delay = 0
        if job.start_time > job.arrival_time:
             delay = job.start_time - job.arrival_time

        if job.job_type == 'training':
            ideal_completion_time = job.arrival_time + job.base_duration
            actual_duration = job.completion_time - job.arrival_time
            perf_factor = actual_duration / job.base_duration if job.base_duration > 0 else 0
            
            # Log the FINAL number of GPUs used
            log_entry = (f"{job.id},{job.arrival_time},{job.start_time},{delay:.2f},{job.base_duration},"
                         f"{ideal_completion_time},{job.completion_time},{perf_factor:.4f},{len(freed_gpus)}\n")
            self.training_log_file.write(log_entry)

        if job.job_type == 'llm_inference':
            for gpu in freed_gpus:
                if gpu.is_llm_server and not gpu.running_tasks:
                    print(f"庁 Clock {self.clock.current_time}: Reverting empty LLM server {gpu.gpu_id} to regular GPU.")
                    gpu.revert_from_llm_server()
    
    def _log_average_inference_delay(self):
        """Calculates and logs the average inference delay for the completed interval."""
        if not self.current_inference_delays:
            avg_delay = 0
            job_count = 0
        else:
            avg_delay = sum(self.current_inference_delays) / len(self.current_inference_delays)
            job_count = len(self.current_inference_delays)

        log_entry = f"{self.clock.current_time},{avg_delay:.2f},{job_count}\n"
        self.inference_delay_log_file.write(log_entry)
        self.current_inference_delays = []

    def run_simulation(self):
        """Main simulation loop with adaptive policies and fair dispatching."""
        if not self.pending_jobs: 
            print("No jobs to simulate.")
            self.print_results()
            return

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
        self.next_delay_log_time = ( (effective_start_time // self.delay_log_interval) + 1 ) * self.delay_log_interval
        
        self.jobs_to_retry = deque() 

        while self.pending_jobs or self.running_jobs or self.jobs_to_retry:
            if self.end_time != -1 and self.clock.current_time >= self.end_time:
                print(f"\n尅 Simulation ended at specified end time: {self.end_time}")
                break
            
            self.clock.tick()

            # --- Elastic Scaling Check ---
            # Try to give more GPUs to running training jobs if available
            self._try_scale_up_training_jobs()

            if self.clock.current_time >= self.next_delay_log_time:
                self._log_average_inference_delay()
                self.next_delay_log_time += self.delay_log_interval

            if self.clock.current_time > 0:
                if self.clock.current_time % self.log_interval == 0:
                    self._log_gpu_usage()
            
            if self.clock.current_time % self.progress_interval == 0 and (self.running_jobs or self.pending_jobs or self.jobs_to_retry):
                
                num_llm_servers = sum(1 for gpu in self.cluster.inference_gpus if gpu.is_llm_server)
                num_llm_jobs = sum(1 for job in self.running_jobs if job.job_type == 'llm_inference')
                
                print(f"葡 Clock {self.clock.current_time}: "
                      f"Pending={len(self.pending_jobs)}, "
                      f"Retrying={len(self.jobs_to_retry)}, "
                      f"Running={len(self.running_jobs)} (LLM: {num_llm_jobs}), "
                      f"LLM Servers={num_llm_servers}, "
                      f"Completed={len(self.completed_jobs)}")
            
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
            
            finished_this_tick = []
            for job in self.running_jobs:
                job.update_progress(self.clock.tick_duration, self.clock.current_time)
                if job.is_complete():
                    finished_this_tick.append(job)
            
            for job in finished_this_tick:
                self._handle_job_completion(job)


    def print_results(self):
        """Prints a final summary and saves it to simulation_summary.txt."""
        
        print(f"Logging remaining inference delays at final time {self.clock.current_time}...")
        self._log_average_inference_delay()

        self.training_log_file.close()
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
            
            all_inference_jobs = inference_jobs + llm_inference_jobs
            
            avg_training_turnaround = sum(j.turnaround_time for j in training_jobs) / len(training_jobs) if training_jobs else 0
            avg_inference_turnaround = sum(j.turnaround_time for j in all_inference_jobs) / len(all_inference_jobs) if all_inference_jobs else 0
            
            avg_training_delay = sum(j.start_time - j.arrival_time for j in training_jobs if j.start_time != -1) / len(training_jobs) if training_jobs else 0
            avg_inference_delay = sum(j.start_time - j.arrival_time for j in all_inference_jobs if j.start_time != -1) / len(all_inference_jobs) if all_inference_jobs else 0

            lines = [
                "--- Simulation Results ---",
                f"Detailed logs saved to 'training_job_log.csv', 'inference_delay_log.csv', and 'gpu_usage_log.csv'",
                f"Total Jobs Completed: {total_jobs}",
                f"Average Training Job Turnaround: {avg_training_turnaround:.2f} seconds",
                f"Average Inference Job Turnaround: {avg_inference_turnaround:.2f} seconds",
                f"Average Training Job Delay (Queue Time): {avg_training_delay:.2f} seconds",
                f"Average Inference Job Delay (Queue Time): {avg_inference_delay:.2f} seconds",
                "--------------------------"
            ]

            for line in lines:
                print(line)
                summary_file.write(line + "\n")