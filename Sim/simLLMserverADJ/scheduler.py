# file: scheduler.py

import math
from collections import deque
from components import (SimulationClock, Job, GPU, GPU_MEMORY_GB, GPU_UTILIZATION_PERCENT, 
                        PREEMPTION_OVERHEAD, RECLAMATION_OVERHEAD, HIGH_UTIL_THRESHOLD, 
                        LOW_UTIL_THRESHOLD, PARTIAL_LOCK_FRACTION, SHARABLE_GPU_MEM_PENALTY_GB,
                        LLM_POLICY_INTERVAL, LLM_MAX_CONCURRENCY, PREEMPTION_COOLDOWN)

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

        # 2. Count usage in the inference pool with the new logic
        for gpu in self.cluster.inference_gpus:
            
            # --- MODIFIED LOGIC ---
            
            # First, check if it's an LLM server. If so, it's *always*
            # counted as an "inference GPU used", even if it's empty.
            if gpu.is_llm_server:
                inference_gpus_used += 1
                
            # If it's NOT an LLM server, then check if it's busy
            # with other types of jobs (regular inference or borrowed training).
            elif not gpu.is_idle():
                has_native_task = False
                has_borrowed_task = False

                for task in gpu.running_tasks.values():
                    job_type = task['job'].job_type
                    
                    if job_type == 'inference': # No 'llm_inference' needed here
                        has_native_task = True
                    elif job_type == 'training':
                        has_borrowed_task = True

                if has_native_task:
                    inference_gpus_used += 1
                if has_borrowed_task:
                    borrowed_gpus_used += 1
            
            # --- END MODIFIED LOGIC ---
        
        self.usage_log_file.write(f"{self.clock.current_time},{training_gpus_used},{inference_gpus_used},{borrowed_gpus_used}\n")

    def _apply_adaptive_llm_policy(self):
        """
        Periodically adjusts the number of LLM servers based on the CURRENT
        load, preempting training jobs if necessary and handling reclamation
        when scaling down.
        """
        
        print(f"--- ⚙️ Clock {self.clock.current_time}: Running adaptive LLM policy ---")
        
        # 1. Measure current LLM job load (running + waiting to be retried)
        llm_jobs_count = 0
        for job in self.running_jobs:
            if job.job_type == 'llm_inference':
                llm_jobs_count += 1
                
        for job in self.jobs_to_retry:
            if job.job_type == 'llm_inference':
                llm_jobs_count += 1
                
        # 2. Calculate the target number of LLM servers needed
        target_llm_gpus = math.ceil(llm_jobs_count / LLM_MAX_CONCURRENCY) + 1

        # --- ADD THIS LINE FOR DEBUGGING ---
        print(f"    Policy Check: Found {llm_jobs_count} active LLM jobs. Target servers: {target_llm_gpus}")

        # 3. Get the current state of the cluster
        current_llm_gpus = [gpu for gpu in self.cluster.inference_gpus if gpu.is_llm_server]
        
        # 4. Scale up or down to meet the target
        gpus_to_change = target_llm_gpus - len(current_llm_gpus)

        if gpus_to_change > 0: # --- Scale UP (NEW PRIORITY) ---
        
            print(f"    Policy: Scaling UP by {gpus_to_change} servers.")
            num_converted = 0

            # Step A: Find and convert idle NON-SHARABLE GPUs
            candidates_nonsharable = [gpu for gpu in self.cluster.inference_gpus if not gpu.is_llm_server and gpu.is_idle() and not gpu.sharable]
            
            for gpu in candidates_nonsharable:
                if num_converted >= gpus_to_change: break
                gpu.convert_to_llm_server()
                num_converted += 1

            # Step B: If still not enough, preempt training jobs from SHARABLE GPUs
            gpus_still_needed = gpus_to_change - num_converted
            if gpus_still_needed > 0:
                print(f"    Policy: Preempting {gpus_still_needed} training jobs to create more servers.")
                for _ in range(gpus_still_needed):
                    victim_job, victim_gpu = self.cluster.find_preemptible_job(self.clock.current_time)
                    if victim_job and victim_gpu:
                        victim_job.preempt_and_pause(victim_gpu, self.clock.current_time)
                        self.preemption_map[victim_gpu.gpu_id] = victim_job
                        self.preemption_count += 1
                        victim_gpu.convert_to_llm_server()
                        num_converted += 1
                    else:
                        print("    Policy: No more victims to preempt.")
                        break # No more victims

            # Step C: If STILL not enough, convert idle SHARABLE GPUs (last resort)
            gpus_still_needed = gpus_to_change - num_converted
            if gpus_still_needed > 0:
                print(f"    Policy: Using {gpus_still_needed} idle sharable GPUs as last resort.")
                candidates_sharable = [gpu for gpu in self.cluster.inference_gpus if not gpu.is_llm_server and gpu.is_idle() and gpu.sharable]
                for gpu in candidates_sharable:
                    if num_converted >= gpus_to_change: break
                    gpu.convert_to_llm_server()
                    num_converted += 1

        elif gpus_to_change < 0: # --- Scale DOWN ---
            # (Scale-down logic is unchanged)
            print(f"    Policy: Scaling DOWN by {abs(gpus_to_change)} servers.")
            num_to_revert = abs(gpus_to_change)
            
            empty_servers = [gpu for gpu in current_llm_gpus if not gpu.running_tasks]
            busy_servers = [gpu for gpu in current_llm_gpus if gpu.running_tasks]
            
            empty_servers.sort(key=lambda gpu: not gpu.sharable)
            busy_servers.sort(key=lambda gpu: not gpu.sharable)
            
            all_candidates = empty_servers + busy_servers
            gpus_to_revert = all_candidates[:num_to_revert]

            print(f"    Policy: Found {len(empty_servers)} empty servers and {len(busy_servers)} busy servers. Reverting {len(gpus_to_revert)}.")

            for gpu_to_revert in gpus_to_revert:
                if gpu_to_revert.running_tasks:
                    # print(f"    Policy: Forcing eviction from busy server {gpu_to_revert.gpu_id}...")
                    tasks_to_evict = list(gpu_to_revert.running_tasks.values())
                    for task in tasks_to_evict:
                        job = task['job']
                        self.running_jobs.remove(job) 
                        self.jobs_to_retry.append(job) 
                        gpu_to_revert.release_task(job) 
                
                if gpu_to_revert.gpu_id in self.preemption_map:
                    reclaiming_job = self.preemption_map.pop(gpu_to_revert.gpu_id)
                    reverted = gpu_to_revert.revert_from_llm_server()
                    if reverted and reclaiming_job in self.running_jobs:
                        reclaiming_job.reclaim_gpu(gpu_to_revert, self.clock.current_time)
                        self.reclamation_count += 1
                    elif not reverted:
                        self.preemption_map[gpu_to_revert.gpu_id] = reclaiming_job
                else:
                    gpu_to_revert.revert_from_llm_server()
    
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
            
            # --- THIS IS THE FIX ---
            # Explicitly check if the GPU needs conversion
            if not gpu.is_llm_server:
                # This is an idle GPU. Convert it now.
                was_converted = gpu.convert_to_llm_server()
                if not was_converted:
                    continue # Conversion failed (maybe not idle?), skip this GPU
                
                # Now that it's converted, get its slots
                slots_to_fill = gpu.llm_slots_available # This is now LLM_MAX_CONCURRENCY
            else:
                # This is an existing server. Get its remaining slots.
                slots_to_fill = gpu.llm_slots_available
            # --- END FIX ---
                
            # Fill this one GPU with as many jobs as it can take
            for _ in range(slots_to_fill):
                if job_index >= num_jobs_to_assign:
                    break 
                
                job = llm_jobs[job_index]
                job.start_time = self.clock.current_time
                
                # --- THIS IS THE FIX ---
                job.assigned_gpus = [gpu]
                # -----------------------
                
                gpu.assign_llm_task(job) 
                self.running_jobs.append(job)
                
                assigned_count += 1
                job_index += 1

        # Return any jobs that are left over
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
            victim_job, victim_gpu = self.cluster.find_preemptible_job(self.clock.current_time)
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
        victim_job, victim_gpu = self.cluster.find_preemptible_job(self.clock.current_time)
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
                victim_job, victim_gpu = self.cluster.find_preemptible_job(self.clock.current_time)
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
                extra_gpus_to_request = math.floor(gpus_needed * 1)
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
        self.jobs_to_retry = deque() # Initialize the retry queue

        # --- Main Simulation Loop ---
        while self.pending_jobs or self.running_jobs or self.jobs_to_retry:
            if self.end_time != -1 and self.clock.current_time >= self.end_time:
                print(f"\n🛑 Simulation ended at specified end time: {self.end_time}")
                break
            
            self.clock.tick()

            # --- Periodic Policy and Logging Calls (at the start of the tick) ---
            if self.clock.current_time > 0:
                # Call the adaptive LLM policy function periodically
                if self.clock.current_time % LLM_POLICY_INTERVAL == 0:
                    self._apply_adaptive_llm_policy()
                # Log GPU usage
                if self.clock.current_time % self.log_interval == 0:
                    self._log_gpu_usage()
            
            if self.clock.current_time % self.progress_interval == 0 and (self.running_jobs or self.pending_jobs or self.jobs_to_retry):
                
                # --- NEW: Calculate real-time LLM stats ---
                num_llm_servers = sum(1 for gpu in self.cluster.inference_gpus if gpu.is_llm_server)
                num_llm_jobs = sum(1 for job in self.running_jobs if job.job_type == 'llm_inference')
                
                # --- UPDATED: Print statement with new stats ---
                print(f"🕒 Clock {self.clock.current_time}: "
                      f"Pending={len(self.pending_jobs)}, "
                      f"Retrying={len(self.jobs_to_retry)}, "
                      f"Running={len(self.running_jobs)} (LLM: {num_llm_jobs}), "
                      f"LLM Servers={num_llm_servers}, "
                      f"Completed={len(self.completed_jobs)}")
                
                # if len(self.running_jobs) > 0 and num_llm_jobs == 0:
                #         print("    [DEBUG] No LLM jobs found. Running jobs have types:")
                #         for job in self.running_jobs:
                #             print(f"    - Job {job.id}, Type: '{job.job_type}'")
            
            # --- Fairer Dispatch Logic (prevents Head-of-Line Blocking) ---
            # 1. Combine jobs to be attempted this tick, prioritizing retries.
            arrived_jobs = list(self.jobs_to_retry)
            self.jobs_to_retry.clear()
            while self.pending_jobs and self.pending_jobs[0].arrival_time <= self.clock.current_time:
                arrived_jobs.append(self.pending_jobs.popleft())

            if arrived_jobs:
                # 2. Separate LLM jobs for batch processing
                arrived_llm_jobs = [j for j in arrived_jobs if j.job_type == 'llm_inference']
                other_arrived_jobs = [j for j in arrived_jobs if j.job_type != 'llm_inference']
                
                # 3. Process the LLM jobs in a single, efficient batch
                unassigned_llm_jobs = self._batch_dispatch_llm_jobs(arrived_llm_jobs)

                # 4. Process other jobs (training, regular inference) one-by-one
                unassigned_other_jobs = []
                for job in other_arrived_jobs:
                    if not self._dispatch_job(job):
                        unassigned_other_jobs.append(job)

                # 5. Add all unassigned jobs to the retry queue for the next tick.
                all_unassigned = sorted(unassigned_llm_jobs + unassigned_other_jobs, key=lambda j: j.arrival_time)
                self.jobs_to_retry.extend(all_unassigned)
            
            # --- Job Progress and Completion Handling ---
            finished_this_tick = []
            for job in self.running_jobs:
                job.update_progress(self.clock.tick_duration, self.clock.current_time)
                if job.is_complete():
                    finished_this_tick.append(job)
            
            # Handle completions last, removing jobs from the running list.
            for job in finished_this_tick:
                self._handle_job_completion(job)


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
