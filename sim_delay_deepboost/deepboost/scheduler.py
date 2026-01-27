# file: scheduler.py

import math
from collections import deque
from components import (SimulationClock, OVERHEAD_COLD_START, 
                        OVERHEAD_WARM_START, OVERHEAD_RECLAIM, OVERHEAD_AFE_SYNC)

class Scheduler:
    def __init__(self, jobs_list, cluster_manager, tick_duration=1):
        self.pending_jobs = deque(sorted(jobs_list, key=lambda j: j.arrival_time))
        self.running_jobs = []
        self.completed_jobs = []
        self.cluster = cluster_manager
        self.clock = SimulationClock(tick_duration=tick_duration)

        # Logging Intervals
        self.delay_log_interval = 600 # 10 minutes
        self.next_delay_log_time = 0  # Will be set in run_simulation
        self.current_inference_delays = [] # Stores delays for jobs started in the interval

        # Initialize log files with HVD suffix
        self.training_log_file = open("training_job_log_hvd.csv", "w")
        self.inference_delay_log_file = open("inference_delay_log_hvd.csv", "w")
        self.usage_log_file = open("gpu_usage_log_hvd.csv", "w")
        self._initialize_logs()

    def _initialize_logs(self):
        """Writes headers to the log files matching the requested format."""
        self.training_log_file.write("job_id,arrival_time,start_time,delay,base_duration,ideal_completion_time,actual_completion_time,performance_factor,gpus\n")
        self.inference_delay_log_file.write("timestamp,average_delay_seconds,job_count\n")
        self.usage_log_file.write("timestamp,training_gpus_used,inference_gpus_used,borrowed_inference_gpus\n")

    def _log_gpu_usage(self):
        """Logs GPU usage mapping DeepBoot states to the requested categories."""
        training_gpus_used = 0
        inference_gpus_used = 0
        borrowed_gpus_used = 0

        # 1. Dedicated Training Pool
        for gpu in self.cluster.training_gpus:
            if gpu.running_tasks: # If not empty, it's used
                training_gpus_used += 1

        # 2. Inference Pool (Mixed States)
        for gpu in self.cluster.inference_gpus:
            if gpu.state in ['RUN', 'PROTECT']:
                # 'RUN' means serving inference. 'PROTECT' means reserved/warm.
                # Both count as "Inference Usage" in DeepBoot context.
                inference_gpus_used += 1
            elif gpu.state == 'TRAIN':
                # 'TRAIN' means loaned to training job
                borrowed_gpus_used += 1
        
        self.usage_log_file.write(f"{self.clock.current_time},{training_gpus_used},{inference_gpus_used},{borrowed_gpus_used}\n")

    def _log_average_inference_delay(self):
        """Calculates and logs the average inference delay for the interval."""
        if not self.current_inference_delays:
            avg_delay = 0
            job_count = 0
        else:
            avg_delay = sum(self.current_inference_delays) / len(self.current_inference_delays)
            job_count = len(self.current_inference_delays)

        log_entry = f"{self.clock.current_time},{avg_delay:.2f},{job_count}\n"
        self.inference_delay_log_file.write(log_entry)
        self.current_inference_delays = []

    def _record_inference_start(self, job):
        """Helper to record delay when an inference job starts."""
        delay = max(0, job.start_time - job.arrival_time)
        self.current_inference_delays.append(delay)

    # --- Dispatch Logic ---

    def _dispatch_llm_inference_job(self, job):
        # 1. Warm Start
        gpu = self.cluster.find_warm_inference_gpu()
        if gpu:
            if gpu.state == 'PROTECT':
                gpu.usage_count += 1
            job.overhead_remaining = OVERHEAD_WARM_START
            self._start_inference(job, gpu)
            return True

        # 2. Cold Start
        gpu = self.cluster.find_free_inference_gpu()
        if gpu:
            gpu.usage_count = 0
            job.overhead_remaining = OVERHEAD_COLD_START
            self._start_inference(job, gpu)
            return True

        # 3. Reclaim
        gpu = self.cluster.find_reclaim_target()
        if gpu:
            gpu.usage_count = 0
            job.overhead_remaining = OVERHEAD_RECLAIM
            if gpu.running_tasks:
                training_job = list(gpu.running_tasks.values())[0]['job']
                self._preempt_training_job(training_job, gpu)
            self._start_inference(job, gpu)
            return True
        
        return False

    def _start_inference(self, job, gpu):
        job.assigned_gpus = [gpu]
        job.start_time = self.clock.current_time
        self._record_inference_start(job) # Log the delay
        gpu.assign_task(job)
        self.running_jobs.append(job)

    def _preempt_training_job(self, job, gpu):
        gpu.release_task(job)
        if gpu in job.assigned_gpus:
            job.assigned_gpus.remove(gpu)
        job.overhead_remaining += OVERHEAD_AFE_SYNC

    def _dispatch_training_job_greedy(self, job):
        desired = 4 
        assigned = []

        # 1. Dedicated
        assigned.extend(self.cluster.find_idle_training_gpus(desired))
        
        # 2. Borrow Inference
        needed = desired - len(assigned)
        if needed > 0:
            assigned.extend(self.cluster.find_loanable_inference_gpus(needed))
        
        if len(assigned) > 0:
            job.assigned_gpus = assigned
            job.start_time = self.clock.current_time
            # Note: For training, we log detailed metrics at completion, not start
            
            for gpu in assigned:
                if gpu.gpu_type == 'inference':
                    gpu.state = 'TRAIN' 
                    gpu.protect_time_remaining = 0
                gpu.assign_task(job)
            
            self.running_jobs.append(job)
            return True
        return False

    def _handle_completion(self, job):
        for gpu in job.assigned_gpus:
            gpu.release_task(job)
            
            # Inference Lifecycle
            if job.job_type in ['inference', 'llm_inference'] and gpu.gpu_type == 'inference':
                if len(gpu.running_tasks) == 0:
                    gpu.state = 'PROTECT'
                    gpu.protect_time_remaining = gpu.calculate_protection_time()
                else:
                    gpu.state = 'RUN'
            
            # Training Lifecycle
            if job.job_type == 'training' and gpu.gpu_type == 'inference':
                gpu.state = 'FREE'
                gpu.usage_count = 0 

        job.record_completion(self.clock.current_time)
        self.running_jobs.remove(job)
        self.completed_jobs.append(job)

        # --- Detailed Training Log Write ---
        if job.job_type == 'training':
            delay = max(0, job.start_time - job.arrival_time)
            ideal_completion_time = job.arrival_time + job.base_duration
            actual_duration = job.completion_time - job.arrival_time
            perf_factor = actual_duration / job.base_duration if job.base_duration > 0 else 0
            # Use current GPU count as 'gpus'
            gpu_count = len(job.assigned_gpus) 
            
            log_entry = (f"{job.id},{job.arrival_time},{job.start_time},{delay:.2f},{job.base_duration},"
                         f"{ideal_completion_time},{job.completion_time},{perf_factor:.4f},{gpu_count}\n")
            self.training_log_file.write(log_entry)

    def run_simulation(self):
        print(f"🚀 Starting DeepBoot Simulation (HVD Logging Enabled)...")
        
        # Initialize delay logging time
        if self.pending_jobs:
            start_t = self.pending_jobs[0].arrival_time
            self.clock.current_time = start_t
            self.next_delay_log_time = ((start_t // self.delay_log_interval) + 1) * self.delay_log_interval
        
        while self.pending_jobs or self.running_jobs:
            self.clock.tick()
            
            # 1. Interval Logging
            if self.clock.current_time >= self.next_delay_log_time:
                self._log_average_inference_delay()
                self.next_delay_log_time += self.delay_log_interval
            
            if self.clock.current_time % 100 == 0: # Usage log interval
                self._log_gpu_usage()

            # 2. Lifecycle Updates
            for gpu in self.cluster.inference_gpus:
                gpu.update_lifecycle(self.clock.tick_duration)
            
            # 3. Dispatch
            failed_dispatch_jobs = []
            while self.pending_jobs:
                job = self.pending_jobs[0]
                if job.arrival_time > self.clock.current_time:
                    break
                
                self.pending_jobs.popleft()
                dispatched = False
                
                if job.job_type in ['inference', 'llm_inference']:
                    dispatched = self._dispatch_llm_inference_job(job)
                else:
                    dispatched = self._dispatch_training_job_greedy(job)
                
                if not dispatched:
                    failed_dispatch_jobs.append(job)
            
            self.pending_jobs.extendleft(reversed(failed_dispatch_jobs))

            # 4. Progress
            finished_jobs = []
            for job in self.running_jobs:
                job.update_progress(self.clock.tick_duration, self.clock.current_time)
                if job.is_complete():
                    finished_jobs.append(job)
            
            for job in finished_jobs:
                self._handle_completion(job)
                
            if self.clock.current_time % 1000 == 0:
                print(f"Time {self.clock.current_time}: Running={len(self.running_jobs)} Completed={len(self.completed_jobs)}")

        print(f"✅ Simulation Complete. Total Completed Jobs: {len(self.completed_jobs)}")
        self.print_results()

    def print_results(self):
        # Final log flush
        self._log_average_inference_delay()
        
        self.training_log_file.close()
        self.inference_delay_log_file.close()
        self.usage_log_file.close()

        print("\n--- Final Results ---")
        if not self.completed_jobs:
            print("No jobs completed.")
            return

        train_jobs = [j for j in self.completed_jobs if j.job_type == 'training']
        inf_jobs = [j for j in self.completed_jobs if j.job_type != 'training']
        
        lines = [
            f"Detailed logs saved to 'training_job_log_hvd.csv', 'inference_delay_log_hvd.csv', and 'gpu_usage_log_hvd.csv'",
            f"Total Jobs Completed: {len(self.completed_jobs)}"
        ]
        
        if train_jobs:
            avg_train = sum(j.turnaround_time for j in train_jobs) / len(train_jobs)
            avg_train_delay = sum(j.start_time - j.arrival_time for j in train_jobs) / len(train_jobs)
            lines.append(f"Average Training Job Turnaround: {avg_train:.2f} seconds")
            lines.append(f"Average Training Job Delay (Queue): {avg_train_delay:.2f} seconds")
            
        if inf_jobs:
            avg_inf = sum(j.turnaround_time for j in inf_jobs) / len(inf_jobs)
            avg_inf_delay = sum(j.start_time - j.arrival_time for j in inf_jobs) / len(inf_jobs)
            lines.append(f"Average Inference Job Turnaround: {avg_inf:.2f} seconds")
            lines.append(f"Average Inference Job Delay (Queue): {avg_inf_delay:.2f} seconds")

        for line in lines:
            print(line)