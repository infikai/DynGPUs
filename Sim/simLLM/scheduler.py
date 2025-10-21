# file: scheduler.py

import math
from collections import deque
from components import (SimulationClock, Job, GPU, GPU_MEMORY_GB, GPU_UTILIZATION_PERCENT, 
                        PREEMPTION_OVERHEAD, RECLAMATION_OVERHEAD, HIGH_UTIL_THRESHOLD, 
                        LOW_UTIL_THRESHOLD, PARTIAL_LOCK_FRACTION, SHARABLE_GPU_MEM_PENALTY_GB)

class Scheduler:
    """
    Manages job scheduling with preemption, reclamation, dynamic locking, 
    and other advanced policies for a heterogeneous GPU cluster.
    """
    def __init__(self, jobs_list, cluster_manager, progress_interval, log_interval, policy_interval, start_time, end_time, tick_duration):
        self.pending_jobs = deque(sorted(jobs_list, key=lambda j: j.arrival_time))
        self.running_jobs = []
        self.completed_jobs = []
        self.cluster = cluster_manager
        self.clock = SimulationClock(tick_duration=tick_duration)
        self.preemption_count = 0
        self.reclamation_count = 0
        self.preemption_map = {}
        self.max_locked_gpus = 0

        # Intervals controlled by user input
        self.progress_interval = progress_interval
        self.log_interval = log_interval
        self.policy_interval = policy_interval
        self.start_time = start_time
        self.end_time = end_time

        # Initialize log files
        self.training_log_file = open("training_job_log.csv", "w")
        self.usage_log_file = open("gpu_usage_log.csv", "w")
        self._initialize_logs()

    def _initialize_logs(self):
        """Writes headers to the log files."""
        self.training_log_file.write("job_id,arrival_time,base_duration,ideal_completion_time,actual_completion_time,performance_factor\n")
        self.usage_log_file.write("timestamp,training_gpus_used,inference_gpus_used,borrowed_inference_gpus\n")

    def _log_gpu_usage(self):
        """Logs a snapshot of GPU usage, separating counts by GPU pool."""
        training_gpus_used = set()
        inference_gpus_used = set()
        borrowed_gpus_used = set()

        for job in self.running_jobs:
            for gpu in job.assigned_gpus:
                if job.job_type == 'training':
                    # ** MODIFIED: Only count GPUs from the training pool here. **
                    if gpu.gpu_type == 'training':
                        training_gpus_used.add(gpu.gpu_id)
                    # Count borrowed GPUs separately.
                    elif gpu.gpu_type == 'inference':
                        borrowed_gpus_used.add(gpu.gpu_id)
                elif job.job_type == 'inference' or job.job_type == 'llm_inference':
                    # This adds any inference GPU running a regular or LLM inference job.
                    inference_gpus_used.add(gpu.gpu_id)
        
        self.usage_log_file.write(f"{self.clock.current_time},{len(training_gpus_used)},{len(inference_gpus_used)},{len(borrowed_gpus_used)}\n")

        
    def _dispatch_job(self, job):
        """Routes a job to the appropriate dispatch method based on its type."""
        if job.job_type == 'training':
            return self._dispatch_training_job(job)
        elif job.job_type == 'inference':
            return self._dispatch_inference_job(job)
        # NEW: Route for the new job type
        elif job.job_type == 'llm_inference':
            return self._dispatch_llm_inference_job(job)
        return False
    
    def _batch_dispatch_llm_jobs(self, llm_jobs):
        """Dispatches a batch of LLM jobs efficiently."""
        if not llm_jobs:
            return []

        num_jobs = len(llm_jobs)
        
        # 1. Get all possible GPU slots from active servers, idle GPUs, or by preempting.
        available_slots, victims_to_preempt = self.cluster.find_resources_for_llm_batch(num_jobs)
        
        # 2. Perform all necessary preemptions found in the previous step.
        if victims_to_preempt:
            gpus_preempted = set()
            for victim_job, victim_gpu in victims_to_preempt:
                if victim_gpu.gpu_id not in gpus_preempted:
                    victim_job.preempt_and_pause(victim_gpu, self.clock.current_time)
                    self.preemption_map[victim_gpu.gpu_id] = victim_job
                    self.preemption_count += 1
                    gpus_preempted.add(victim_gpu.gpu_id)

        # 3. Assign jobs to the pool of available slots.
        assigned_count = 0
        for i, job in enumerate(llm_jobs):
            if i < len(available_slots):
                gpu = available_slots[i]
                job.assigned_gpus = [gpu]
                job.start_time = self.clock.current_time
                gpu.assign_llm_task(job)
                self.running_jobs.append(job)
                assigned_count += 1
            else:
                # Stop if we run out of slots.
                break 
    
        # 4. Return any jobs that couldn't be scheduled.
        unassigned_jobs = llm_jobs[assigned_count:]
        return unassigned_jobs

    def _dispatch_llm_inference_job(self, job):
        """
        Schedules a single LLM inference request, using preemption if necessary.
        This function is kept for consistency, but the main loop uses the batch dispatcher.
        """
        # 1. Use the corrected finder to get a GPU that is either an
        #    active LLM server with slots or an idle GPU ready for conversion.
        gpu = self.cluster.find_gpu_for_llm_job()
        
        # 2. If no GPU is readily available, try to free one up via preemption.
        if not gpu:
            victim_job, victim_gpu = self.cluster.find_preemptible_job()
            if victim_job and victim_gpu:
                # Preempting the training job makes its GPU idle and thus convertible.
                victim_job.preempt_and_pause(victim_gpu, self.clock.current_time)
                self.preemption_map[victim_gpu.gpu_id] = victim_job
                self.preemption_count += 1
                
                # The newly freed GPU is our target.
                gpu = victim_gpu
        
        # 3. If we have a GPU (either found or freed), assign the job.
        if gpu:
            job.assigned_gpus = [gpu]
            job.start_time = self.clock.current_time
            
            # The assign_llm_task method handles the conversion to an exclusive-use server.
            gpu.assign_llm_task(job)
            
            self.running_jobs.append(job)
            return True
            
        # 4. If no GPU could be found or freed, the job must wait.
        return False

    def _dispatch_inference_job(self, job):
        """Routes an inference job to the correct scheduler based on its size."""
        is_large_job = (job.memory_required > (GPU_MEMORY_GB - SHARABLE_GPU_MEM_PENALTY_GB) or 
                        job.utilization_required > GPU_UTILIZATION_PERCENT)
        
        if is_large_job:
            return self._dispatch_large_inference_job(job)
        else:
            return self._dispatch_stackable_inference_job(job)

    def _dispatch_stackable_inference_job(self, job):
        """Schedules a small inference job, using preemption for a single slot if needed."""
        job.gpus_needed = 1 
        gpu = self.cluster.find_gpu_for_stackable_inference(job)
        if gpu:
            job.assign_resources([gpu], self.clock.current_time)
            self.running_jobs.append(job)
            return True

        # If no space, try to preempt a training job
        victim_job, victim_gpu = self.cluster.find_preemptible_job()
        if victim_job and victim_gpu:
            victim_job.preempt_and_pause(victim_gpu, self.clock.current_time)
            self.preemption_map[victim_gpu.gpu_id] = victim_job
            self.preemption_count += 1
            
            if victim_gpu.can_fit(job):
                job.assign_resources([victim_gpu], self.clock.current_time)
                self.running_jobs.append(job)
                return True
        return False

    def _dispatch_large_inference_job(self, job):
        """Schedules a large inference job, using preemption to secure multiple slots if needed."""
        effective_gpu_mem = GPU_MEMORY_GB - SHARABLE_GPU_MEM_PENALTY_GB
        gpus_needed = max(math.ceil(job.memory_required / effective_gpu_mem),
                          math.ceil(job.utilization_required / GPU_UTILIZATION_PERCENT), 1)
        job.gpus_needed = gpus_needed
        
        allocated_gpus = self.cluster.find_idle_gpus_in_inference_pool(gpus_needed)
        
        gpus_still_needed = gpus_needed - len(allocated_gpus)

        if gpus_still_needed > 0:
            preempted_gpus = []
            for _ in range(gpus_still_needed):
                victim_job, victim_gpu = self.cluster.find_preemptible_job()
                # Ensure we don't try to preempt from a GPU we just allocated as idle
                if victim_job and victim_gpu and victim_gpu not in allocated_gpus:
                    victim_job.preempt_and_pause(victim_gpu, self.clock.current_time)
                    self.preemption_map[victim_gpu.gpu_id] = victim_job
                    self.preemption_count += 1
                    preempted_gpus.append(victim_gpu)
                else:
                    # Rollback any preemptions made in this failed attempt
                    for gpu in preempted_gpus:
                         reclaiming_job = self.preemption_map.pop(gpu.gpu_id)
                         reclaiming_job.reclaim_gpu(gpu, self.clock.current_time)
                         self.preemption_count -= 1
                    return False
            allocated_gpus.extend(preempted_gpus)

        if len(allocated_gpus) == gpus_needed:
            job.assign_resources(allocated_gpus, self.clock.current_time)
            self.running_jobs.append(job)
            return True
        return False
        
    def _dispatch_training_job(self, job):
        """Schedules a training job, requiring it to fit entirely in the training pool."""
        gpus_needed = max(math.ceil(job.memory_required / GPU_MEMORY_GB),
                          math.ceil(job.utilization_required / GPU_UTILIZATION_PERCENT), 1)
        job.gpus_needed = gpus_needed

        # ** MODIFIED: A training job's base requirement must be met *only* from the training pool. **
        allocated_gpus = self.cluster.find_idle_gpus_for_training(gpus_needed)
        
        # If the base requirement is met by the training pool, proceed to schedule.
        if len(allocated_gpus) == gpus_needed:
            # Check if this job is eligible for greedy speedup from the inference pool.
            mem_slice_per_gpu = job.memory_required / gpus_needed
            effective_gpu_mem = GPU_MEMORY_GB - SHARABLE_GPU_MEM_PENALTY_GB
            is_high_memory_job = mem_slice_per_gpu > effective_gpu_mem

            if not is_high_memory_job:
                extra_gpus_to_request = math.floor(gpus_needed * 0.4)
                if extra_gpus_to_request > 0:
                    # The inference pool is now only used for these extra GPUs.
                    extra_gpus = self.cluster.find_idle_borrowable_gpus(extra_gpus_to_request)
                    if extra_gpus: 
                        allocated_gpus.extend(extra_gpus)
            
            job.assign_resources(allocated_gpus, self.clock.current_time)
            self.running_jobs.append(job)
            return True
            
        # If the training pool did not have enough GPUs, the job cannot be scheduled and must wait.
        return False
         
    def _handle_job_completion(self, job):
        """Processes a finished job, logs training data, and handles reclamation."""
        freed_gpus = list(job.assigned_gpus)
        self.cluster.release_resources_for_job(job)
        job.record_completion(self.clock.current_time)
        self.running_jobs.remove(job)
        self.completed_jobs.append(job)

        if job.job_type == 'training':
            ideal_completion_time = job.arrival_time + job.base_duration
            actual_duration = job.completion_time - job.arrival_time
            perf_factor = actual_duration / job.base_duration if job.base_duration > 0 else 0
            log_entry = (f"{job.id},{job.arrival_time},{job.base_duration},"
                         f"{ideal_completion_time},{job.completion_time},{perf_factor:.4f}\n")
            self.training_log_file.write(log_entry)

        if job.job_type == 'inference' or job.job_type == 'llm_inference':
            for gpu in freed_gpus:
                if gpu.gpu_id in self.preemption_map:
                    # ** This existing logic will now correctly handle LLM jobs **
                    if gpu.is_idle():
                        reclaiming_job = self.preemption_map[gpu.gpu_id]
                        if reclaiming_job in self.running_jobs:
                            reclaiming_job.reclaim_gpu(gpu, self.clock.current_time)
                            self.reclamation_count += 1
                            print(f"✅ Clock {self.clock.current_time}: RECLAIMED GPU {gpu.gpu_id} for training job {reclaiming_job.id}.")
                        del self.preemption_map[gpu.gpu_id]
    
    def run_simulation(self):
        """Main simulation loop with smart start time logic."""
        if not self.pending_jobs: 
            print("No jobs to simulate.")
            self.print_results()
            return

        # ** NEW: Determine the effective start time **
        effective_start_time = self.start_time
        if self.start_time == 0:
            # If default start time is used, fast-forward to the first job's arrival
            effective_start_time = self.pending_jobs[0].arrival_time
            print(f"No specific start time given. Fast-forwarding to first job arrival at time {effective_start_time}.")

        # Filter jobs that arrive before the effective start time
        # ** CORRECTED: Ensure the result of the filter is converted back to a deque **
        original_job_count = len(self.pending_jobs)
        filtered_list = [j for j in self.pending_jobs if j.arrival_time >= effective_start_time]
        self.pending_jobs = deque(filtered_list)
        print(f"Filtered out {original_job_count - len(self.pending_jobs)} jobs that arrived before effective start time {effective_start_time}.")
        
        if not self.pending_jobs: 
            print("No jobs to simulate in the specified time window.")
            self.print_results()
            return

        # Set the clock's initial time
        self.clock.current_time = effective_start_time

        while self.pending_jobs or self.running_jobs:
            if self.end_time != -1 and self.clock.current_time >= self.end_time:
                print(f"\n🛑 Simulation ended at specified end time: {self.end_time}")
                break
            
            self.clock.tick()

            if self.clock.current_time > 0:
                if self.clock.current_time % self.log_interval == 0:
                    self._log_gpu_usage()
            
            # # ** MODIFIED: New, highly efficient loop for dispatching arrived jobs **
            # # It only checks the front of the queue, since it's sorted.
            # while self.pending_jobs and self.pending_jobs[0].arrival_time <= self.clock.current_time:
            #     job_to_dispatch = self.pending_jobs.popleft() # O(1) operation
            #     if not self._dispatch_job(job_to_dispatch):
            #         # If dispatch fails (e.g., cluster is full), put the job back at the front
            #         # and stop trying to dispatch jobs for this time tick.
            #         self.pending_jobs.appendleft(job_to_dispatch)
            #         break

            # ** MODIFIED: New dispatch loop to prevent head-of-line blocking **
            arrived_jobs = []
            while self.pending_jobs and self.pending_jobs[0].arrival_time <= self.clock.current_time:
                arrived_jobs.append(self.pending_jobs.popleft())

            if arrived_jobs:
                # 2. Separate LLM jobs from all other types
                arrived_llm_jobs = [j for j in arrived_jobs if j.job_type == 'llm_inference']
                other_arrived_jobs = [j for j in arrived_jobs if j.job_type != 'llm_inference']
                
                # 3. Process the LLM jobs in a single, efficient batch
                unassigned_llm_jobs = self._batch_dispatch_llm_jobs(arrived_llm_jobs)

                # 4. Process other jobs (training, regular inference) one-by-one
                unassigned_other_jobs = []
                for job in other_arrived_jobs:
                    if not self._dispatch_job(job):
                        unassigned_other_jobs.append(job)

                # 5. Add any jobs that couldn't be scheduled back to the front of the queue
                all_unassigned = sorted(unassigned_llm_jobs + unassigned_other_jobs, key=lambda j: j.arrival_time, reverse=True)
                if all_unassigned:
                    for job in all_unassigned:
                        self.pending_jobs.appendleft(job)

            finished_this_tick = []
            for job in self.running_jobs:
                job.update_progress(self.clock.tick_duration, self.clock.current_time)
                if job.is_complete():
                    finished_this_tick.append(job)
            
            for job in finished_this_tick:
                self._handle_job_completion(job)

            if self.clock.current_time % self.progress_interval == 0 and (self.running_jobs or self.pending_jobs):
                print(f"🕒 Clock {self.clock.current_time}: Pending={len(self.pending_jobs)}, Running={len(self.running_jobs)}, Completed={len(self.completed_jobs)}")

    def print_results(self):
        """Prints a final summary and saves it to simulation_summary.txt."""
        self.training_log_file.close()
        self.usage_log_file.close()

        # ** NEW: Open a file to log the summary output **
        with open("simulation_summary.txt", "w") as summary_file:
            total_jobs = len(self.completed_jobs)
            if total_jobs == 0:
                summary_text = "No jobs were completed in the simulation window."
                print(summary_text)
                summary_file.write(summary_text + "\n")
                return

            training_jobs = [j for j in self.completed_jobs if j.job_type == 'training']
            inference_jobs = [j for j in self.completed_jobs if j.job_type == 'inference']
            avg_training_turnaround = sum(j.turnaround_time for j in training_jobs) / len(training_jobs) if training_jobs else 0
            avg_inference_turnaround = sum(j.turnaround_time for j in inference_jobs) / len(inference_jobs) if inference_jobs else 0
            
            # Build the output lines
            lines = [
                "--- Simulation Results ---",
                f"Detailed logs saved to 'training_job_log.csv' and 'gpu_usage_log.csv'",
                f"Total Jobs Completed: {total_jobs}",
                f"Total Preemptions: {self.preemption_count}",
                f"Total Successful Reclamations: {self.reclamation_count}",
                f"Peak number of sharable GPUs locked by policy: {self.max_locked_gpus}",
                f"Average Training Job Turnaround: {avg_training_turnaround:.2f} seconds",
                f"Average Inference Job Turnaround: {avg_inference_turnaround:.2f} seconds",
                "--------------------------"
            ]

            # Print to console and write to the summary file
            for line in lines:
                print(line)
                summary_file.write(line + "\n")
