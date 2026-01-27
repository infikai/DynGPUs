# file: scheduler.py

import math
from collections import deque
from components import (SimulationClock, Job, GPU, GPU_MEMORY_GB, GPU_UTILIZATION_PERCENT, 
                        LLM_MAX_CONCURRENCY)

class Scheduler:
    def __init__(self, jobs_list, cluster_manager, progress_interval, log_interval, start_time, end_time, tick_duration):
        self.pending_jobs = deque(sorted(jobs_list, key=lambda j: j.arrival_time))
        self.running_jobs = []
        self.completed_jobs = []
        self.cluster = cluster_manager
        self.clock = SimulationClock(tick_duration=tick_duration)

        # Intervals
        self.progress_interval = progress_interval
        self.log_interval = log_interval
        self.start_time = start_time
        self.end_time = end_time

        # Logging attributes
        self.delay_log_interval = 600
        self.next_delay_log_time = 0
        self.current_inference_delays = []

        # Initialize logs
        self.training_log_file = open("training_job_log_hvd.csv", "w")
        self.inference_delay_log_file = open("inference_delay_log_hvd.csv", "w")
        self.usage_log_file = open("gpu_usage_log_hvd.csv", "w")
        self._initialize_logs()

    def _initialize_logs(self):
        self.training_log_file.write("job_id,arrival_time,start_time,delay,base_duration,ideal_completion_time,actual_completion_time,performance_factor,gpus\n")
        self.inference_delay_log_file.write("timestamp,average_delay_seconds,job_count\n")
        self.usage_log_file.write("timestamp,training_gpus_used,inference_gpus_used,borrowed_inference_gpus\n")

    def _log_gpu_usage(self):
        training_gpus_used = 0
        inference_gpus_used = 0
        borrowed_gpus_used = 0 

        for gpu in self.cluster.training_gpus:
            if not gpu.is_idle():
                training_gpus_used += 1

        for gpu in self.cluster.inference_gpus:
            if gpu.is_llm_server:
                inference_gpus_used += 1
            elif not gpu.is_idle():
                inference_gpus_used += 1
                for task in gpu.running_tasks.values():
                    if task['job'].job_type == 'training':
                         borrowed_gpus_used += 1
                         inference_gpus_used -= 1
                         break
        
        self.usage_log_file.write(f"{self.clock.current_time},{training_gpus_used},{inference_gpus_used},{borrowed_gpus_used}\n")
    
    def _dispatch_job(self, job):
        if job.job_type == 'training':
            return self._dispatch_training_job(job)
        elif job.job_type == 'inference':
            return self._dispatch_inference_job(job)
        elif job.job_type == 'llm_inference':
            return self._dispatch_llm_inference_job(job)
        return False
    
    def _batch_dispatch_llm_jobs(self, llm_jobs):
        if not llm_jobs:
            return []

        num_jobs_to_assign = len(llm_jobs)
        # This call was failing previously; now fixed in cluster_manager.py
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

        return llm_jobs[assigned_count:]

    def _dispatch_llm_inference_job(self, job):
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
        is_large_job = (job.memory_required > GPU_MEMORY_GB or 
                        job.utilization_required > GPU_UTILIZATION_PERCENT)
        if is_large_job:
            return self._dispatch_large_inference_job(job)
        return self._dispatch_stackable_inference_job(job)

    def _dispatch_stackable_inference_job(self, job):
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
        # 1. Calculate ideal resources needed
        desired_gpus = max(math.ceil(job.memory_required / GPU_MEMORY_GB),
                           math.ceil(job.utilization_required / GPU_UTILIZATION_PERCENT), 1)
        job.gpus_needed = desired_gpus
        min_gpus = 1 

        assigned_gpus = []

        # 2. Priority 1: Dedicated Training GPUs
        for gpu in self.cluster.training_gpus:
            if gpu.is_idle():
                assigned_gpus.append(gpu)
                if len(assigned_gpus) == desired_gpus:
                    break
        
        # 3. Priority 2: Borrow from Inference Pool
        if len(assigned_gpus) < desired_gpus:
            for gpu in self.cluster.inference_gpus:
                if gpu.is_idle():
                    assigned_gpus.append(gpu)
                    if len(assigned_gpus) == desired_gpus:
                        break
        
        # 4. Dispatch if MINIMUM requirement met
        if len(assigned_gpus) >= min_gpus:
            for gpu in assigned_gpus:
                if gpu.gpu_type == 'inference':
                    gpu.state = 'TRAIN'
                    gpu.protect_time_remaining = 0

            job.assign_resources(assigned_gpus, self.clock.current_time)
            self.running_jobs.append(job)
            return True
        return False

    def _try_scale_up_training_jobs(self):
        for job in self.running_jobs:
            if job.job_type == 'training':
                current_count = len(job.assigned_gpus)
                desired = getattr(job, 'gpus_needed', 1) 
                
                if current_count < desired:
                    needed = desired - current_count
                    newly_acquired = []

                    for gpu in self.cluster.training_gpus:
                        if gpu.is_idle():
                            newly_acquired.append(gpu)
                            if len(newly_acquired) == needed: break
                    
                    if len(newly_acquired) < needed:
                        remaining = needed - len(newly_acquired)
                        for gpu in self.cluster.inference_gpus:
                            if gpu.is_idle():
                                newly_acquired.append(gpu)
                                if len(newly_acquired) >= remaining: break
                        if len(newly_acquired) > needed:
                            newly_acquired = newly_acquired[:needed]

                    if newly_acquired:
                        old_gpus = list(job.assigned_gpus)
                        for gpu in old_gpus: gpu.release_task(job)
                        
                        all_gpus = old_gpus + newly_acquired
                        for gpu in newly_acquired:
                             if gpu.gpu_type == 'inference':
                                gpu.state = 'TRAIN'
                                gpu.protect_time_remaining = 0
                        
                        job.assign_resources(all_gpus, job.start_time)

    def _handle_job_completion(self, job):
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
            
            log_entry = (f"{job.id},{job.arrival_time},{job.start_time},{delay:.2f},{job.base_duration},"
                         f"{ideal_completion_time},{job.completion_time},{perf_factor:.4f},{len(freed_gpus)}\n")
            self.training_log_file.write(log_entry)

        if job.job_type == 'llm_inference':
            for gpu in freed_gpus:
                if gpu.is_llm_server and not gpu.running_tasks:
                    print(f"庁 Clock {self.clock.current_time}: Reverting empty LLM server {gpu.gpu_id} to regular GPU.")
                    gpu.revert_from_llm_server()
    
    def _log_average_inference_delay(self):
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
        if not self.pending_jobs: 
            print("No jobs to simulate.")
            self.print_results()
            return

        effective_start_time = self.start_time
        if self.start_time == 0:
            effective_start_time = self.pending_jobs[0].arrival_time
            print(f"Fast-forwarding to time {effective_start_time}.")

        # Fast forward pending jobs
        while self.pending_jobs and self.pending_jobs[0].arrival_time < effective_start_time:
             self.pending_jobs.popleft()
        
        if not self.pending_jobs: 
            print("No jobs left in window.")
            self.print_results()
            return

        self.clock.current_time = effective_start_time
        self.next_delay_log_time = ( (effective_start_time // self.delay_log_interval) + 1 ) * self.delay_log_interval
        self.jobs_to_retry = deque() 

        while self.pending_jobs or self.running_jobs or self.jobs_to_retry:
            if self.end_time != -1 and self.clock.current_time >= self.end_time:
                break
            
            self.clock.tick()
            self._try_scale_up_training_jobs()

            if self.clock.current_time >= self.next_delay_log_time:
                self._log_average_inference_delay()
                self.next_delay_log_time += self.delay_log_interval

            if self.clock.current_time > 0 and self.clock.current_time % self.log_interval == 0:
                self._log_gpu_usage()
            
            if self.clock.current_time % self.progress_interval == 0:
                num_llm = sum(1 for j in self.running_jobs if j.job_type=='llm_inference')
                print(f"Clock {self.clock.current_time}: Pending={len(self.pending_jobs)}, Running={len(self.running_jobs)} (LLM:{num_llm}), Completed={len(self.completed_jobs)}")
            
            arrived_jobs = list(self.jobs_to_retry)
            self.jobs_to_retry.clear()
            while self.pending_jobs and self.pending_jobs[0].arrival_time <= self.clock.current_time:
                arrived_jobs.append(self.pending_jobs.popleft())

            if arrived_jobs:
                arrived_llm_jobs = [j for j in arrived_jobs if j.job_type == 'llm_inference']
                other_jobs = [j for j in arrived_jobs if j.job_type != 'llm_inference']
                
                unassigned_llm = self._batch_dispatch_llm_jobs(arrived_llm_jobs)
                
                unassigned_other = []
                for job in other_jobs:
                    if not self._dispatch_job(job):
                        unassigned_other.append(job)

                retry_list = unassigned_llm + unassigned_other
                retry_list.sort(key=lambda j: j.arrival_time)
                self.jobs_to_retry.extend(retry_list)
            
            finished = []
            for job in self.running_jobs:
                job.update_progress(self.clock.tick_duration, self.clock.current_time)
                if job.is_complete(): finished.append(job)
            
            for job in finished: self._handle_job_completion(job)

    def print_results(self):
        self._log_average_inference_delay()
        self.training_log_file.close()
        self.inference_delay_log_file.close()
        self.usage_log_file.close()
        print("Simulation Complete. Results saved.")