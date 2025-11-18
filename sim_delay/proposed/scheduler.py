# file: scheduler.py

import math
from collections import deque
from components import (SimulationClock, Job, GPU, GPU_MEMORY_GB, GPU_UTILIZATION_PERCENT, 
                        PREEMPTION_OVERHEAD, RECLAMATION_OVERHEAD, 
                        LLM_POLICY_INTERVAL, LLM_MAX_CONCURRENCY, PREEMPTION_COOLDOWN)

class Scheduler:
    def __init__(self, jobs_list, cluster_manager, progress_interval, log_interval, start_time, end_time, tick_duration, end_time_threshold):
        self.pending_jobs = deque(sorted(jobs_list, key=lambda j: j.arrival_time))
        self.running_jobs = []
        self.completed_jobs = []
        self.cluster = cluster_manager
        self.clock = SimulationClock(tick_duration=tick_duration)
        self.preemption_count = 0
        self.reclamation_count = 0
        
        # Maps GPU ID -> Job object waiting for it
        self.preemption_map = {} 

        self.progress_interval = progress_interval
        self.log_interval = log_interval
        self.start_time = start_time
        self.end_time = end_time
        self.end_time_threshold = end_time_threshold

        self.delay_log_interval = 600 
        self.next_delay_log_time = 0 
        self.current_inference_delays = [] 

        self.training_log_file = open("training_job_log.csv", "w")
        self.usage_log_file = open("gpu_usage_log.csv", "w")
        self.inference_delay_log_file = open("inference_delay_log.csv", "w")
        self._initialize_logs()

    def _initialize_logs(self):
        self.training_log_file.write("job_id,arrival_time,start_time,delay,base_duration,ideal_completion_time,actual_completion_time,performance_factor,gpus\n")
        self.inference_delay_log_file.write("timestamp,average_delay_seconds,job_count\n")
        self.usage_log_file.write("timestamp,train_pool_used,infer_pool_used_by_train,infer_pool_active_llm,train_pool_active_llm\n")

    def _log_gpu_usage(self):
        train_pool_used = sum(1 for g in self.cluster.training_pool if not g.is_idle() and not g.is_llm_server)
        infer_pool_borrowed = sum(1 for g in self.cluster.inference_pool 
                                  if not g.is_llm_server and not g.is_idle())
        infer_pool_llm = sum(1 for g in self.cluster.inference_pool 
                             if g.is_llm_server and g.running_tasks)
        train_pool_llm = sum(1 for g in self.cluster.training_pool 
                             if g.is_llm_server and g.running_tasks)

        self.usage_log_file.write(f"{self.clock.current_time},{train_pool_used},{infer_pool_borrowed},{infer_pool_llm},{train_pool_llm}\n")

    def _log_average_inference_delay(self):
        if not self.current_inference_delays:
            avg_delay = 0
            job_count = 0
        else:
            avg_delay = sum(self.current_inference_delays) / len(self.current_inference_delays)
            job_count = len(self.current_inference_delays)
        self.inference_delay_log_file.write(f"{self.clock.current_time},{avg_delay:.2f},{job_count}\n")
        self.current_inference_delays = []

    def _dispatch_job(self, job):
        if job.job_type == 'training':
            return self._dispatch_training_job(job)
        elif job.job_type == 'llm_inference':
            return self._dispatch_llm_inference_job(job)
        elif job.job_type == 'inference':
             return self._dispatch_training_job(job)
        return False
    
    def _dispatch_llm_inference_job(self, job):
        # P1: Existing Server (Inference Pool)
        gpu = self.cluster.find_gpu_for_llm_job(self.clock.current_time)
        
        # P2: Idle (Inference Pool)
        if not gpu:
            idle_infer_gpus = self.cluster.find_idle_gpus_in_inference_pool()
            if idle_infer_gpus:
                gpu = idle_infer_gpus[0]
                gpu.convert_to_llm_server()

        # P3: Borrow Idle (Training Pool)
        if not gpu:
            idle_train_gpus = self.cluster.find_idle_gpus_in_training_pool()
            if idle_train_gpus:
                gpu = idle_train_gpus[0]
                gpu.convert_to_llm_server()

        # P4: Reclaim (Preempt Squatters in Inference Pool)
        if not gpu:
            victim_job, victim_gpu = self.cluster.find_borrowed_gpu_to_reclaim(self.clock.current_time)
            if victim_job and victim_gpu:
                victim_job.preempt_and_pause(victim_gpu, self.clock.current_time)
                self.preemption_map[victim_gpu.gpu_id] = victim_job
                self.preemption_count += 1
                victim_gpu.convert_to_llm_server(drain_at_time=self.clock.current_time + 100)
                gpu = victim_gpu

        if gpu:
            job.assigned_gpus = [gpu]
            job.start_time = self.clock.current_time
            delay = max(0, job.start_time - job.arrival_time)
            self.current_inference_delays.append(delay)
            gpu.assign_llm_task(job)
            self.running_jobs.append(job)
            return True
        return False 

    def _batch_dispatch_llm_jobs(self, llm_jobs):
        remaining_jobs = []
        for job in llm_jobs:
            if not self._dispatch_llm_inference_job(job):
                remaining_jobs.append(job)
        return remaining_jobs

    def _dispatch_training_job(self, job):
        gpus_needed = max(math.ceil(job.memory_required / GPU_MEMORY_GB),
                          math.ceil(job.utilization_required / GPU_UTILIZATION_PERCENT), 1)
        job.gpus_needed = gpus_needed
        job.max_allowable_duration = job.ideal_duration * self.end_time_threshold

        # 1. Find Strictly Idle Training GPUs (Immediate use)
        #    (Also include empty LLM servers in Training pool - they are effectively idle)
        idle_gpus = []
        for g in self.cluster.training_pool:
            if g.is_idle():
                idle_gpus.append(g)
            elif g.is_llm_server and not g.running_tasks:
                # Immediate reclaim of empty server
                g.revert_from_llm_server()
                idle_gpus.append(g)

        # 2. Find "Occupied" Training GPUs (Running LLM -> Can be drained)
        #    Only consider those NOT already mapped to another job
        occupied_gpus = [g for g in self.cluster.find_non_idle_training_gpus_for_reclamation() 
                         if g.gpu_id not in self.preemption_map]

        total_potential = len(idle_gpus) + len(occupied_gpus)
        
        # We allow starting if we can eventually satisfy the request
        if total_potential >= gpus_needed:
            
            # A. Determine how many we need to drain
            num_to_assign_now = min(len(idle_gpus), gpus_needed)
            num_to_drain = gpus_needed - num_to_assign_now
            
            # B. Assign available idle GPUs
            gpus_to_start = idle_gpus[:num_to_assign_now]
            
            # C. Mark others as draining
            if num_to_drain > 0:
                gpus_draining = occupied_gpus[:num_to_drain]
                for gpu in gpus_draining:
                    gpu.drain_at_time = self.clock.current_time
                    self.preemption_map[gpu.gpu_id] = job
                    # print(f"⚠️ Clock {self.clock.current_time}: Marked {gpu.gpu_id} as draining for Job {job.id}")

            # D. Start job (Partial or Full)
            if gpus_to_start:
                job.assign_resources(gpus_to_start, self.clock.current_time)
                self.running_jobs.append(job)
                # print(f"🚀 Clock {self.clock.current_time}: Started Job {job.id} with {len(gpus_to_start)}/{gpus_needed} GPUs (Waiting for {num_to_drain} to drain).")
                return True
            else:
                # Case: All needed GPUs are currently occupied.
                # We marked them as draining, but we can't "start" the job with 0 GPUs.
                # It stays in pending. As GPUs free up, they will revert and be picked up
                # by this job in subsequent ticks (via idle_gpus check).
                return False

        return False

    def _handle_job_completion(self, job):
        freed_gpus = list(job.assigned_gpus)
        self.cluster.release_resources_for_job(job)
        job.record_completion(self.clock.current_time)
        self.running_jobs.remove(job)
        self.completed_jobs.append(job)

        if job.job_type == 'training':
            ideal_completion_time = job.arrival_time + job.base_duration
            actual_duration = job.completion_time - job.arrival_time
            perf_factor = actual_duration / job.base_duration if job.base_duration > 0 else 0
            delay = max(0, job.start_time - job.arrival_time) if job.start_time != -1 else 0
            log_entry = (f"{job.id},{job.arrival_time},{job.start_time},{delay:.2f},{job.base_duration},"
                         f"{ideal_completion_time},{job.completion_time},{perf_factor:.4f},{job.gpus_needed}\n")
            self.training_log_file.write(log_entry)

        # --- Cleanup & Reclamation Logic ---
        for gpu in freed_gpus:
            
            # 1. Handle Preemption/Reclamation (Job waiting for this specific GPU)
            if gpu.gpu_id in self.preemption_map:
                reclaiming_job = self.preemption_map[gpu.gpu_id]
                
                # If GPU is now free (idle or empty server)
                if gpu.is_idle() or (gpu.is_llm_server and not gpu.running_tasks):
                    
                    # Revert if it was a server
                    if gpu.is_llm_server:
                         gpu.revert_from_llm_server()
                    
                    # Remove from map
                    del self.preemption_map[gpu.gpu_id]
                    
                    # Give to job if it's already running (Partial start case)
                    if reclaiming_job in self.running_jobs:
                        reclaiming_job.reclaim_gpu(gpu, self.clock.current_time)
                        self.reclamation_count += 1
                        # print(f"✅ Clock {self.clock.current_time}: Job {reclaiming_job.id} reclaimed draining GPU {gpu.gpu_id}.")
                    else:
                        # If job is still pending (started with 0 GPUs), it just becomes idle
                        # and will be picked up in the next dispatch loop.
                        pass
            
            # 2. Auto-revert logic for Training Pool GPUs
            # If an LLM job finishes on a Training GPU, and no one is explicitly waiting (map checked above),
            # check if it should revert anyway to be available for general training.
            if job.job_type == 'llm_inference' and gpu.pool_type == 'training':
                if gpu.is_llm_server and not gpu.running_tasks:
                     gpu.revert_from_llm_server()
                     # print(f"Testing: Reverted Training GPU {gpu.gpu_id} from LLM server to Idle.")

    def run_simulation(self):
        if not self.pending_jobs: return

        effective_start_time = self.start_time if self.start_time > 0 else self.pending_jobs[0].arrival_time
        self.clock.current_time = effective_start_time
        self.next_delay_log_time = ( (effective_start_time // self.delay_log_interval) + 1 ) * self.delay_log_interval
        
        self.jobs_to_retry = deque()
        self.pending_jobs = deque([j for j in self.pending_jobs if j.arrival_time >= effective_start_time])

        while self.pending_jobs or self.running_jobs or self.jobs_to_retry:
            if self.end_time != -1 and self.clock.current_time >= self.end_time:
                break
            
            self.clock.tick()
            
            if self.clock.current_time >= self.next_delay_log_time:
                self._log_average_inference_delay()
                self.next_delay_log_time += self.delay_log_interval

            if self.clock.current_time % self.log_interval == 0:
                self._log_gpu_usage()
            
            if self.clock.current_time % self.progress_interval == 0:
                 print(f"🕒 Clock {self.clock.current_time}: Running={len(self.running_jobs)}, Pending={len(self.pending_jobs)}")
            
            arrived_jobs = list(self.jobs_to_retry)
            self.jobs_to_retry.clear()
            while self.pending_jobs and self.pending_jobs[0].arrival_time <= self.clock.current_time:
                arrived_jobs.append(self.pending_jobs.popleft())

            if arrived_jobs:
                arrived_llm_jobs = [j for j in arrived_jobs if j.job_type == 'llm_inference']
                other_arrived_jobs = [j for j in arrived_jobs if j.job_type != 'llm_inference']
                
                unassigned_llm = self._batch_dispatch_llm_jobs(arrived_llm_jobs)
                
                unassigned_other = []
                for job in other_arrived_jobs:
                    if not self._dispatch_job(job):
                        unassigned_other.append(job)

                self.jobs_to_retry.extend(unassigned_llm + unassigned_other)
            
            finished_this_tick = []
            for job in self.running_jobs:
                job.update_progress(self.clock.tick_duration, self.clock.current_time)
                if job.is_complete():
                    finished_this_tick.append(job)
            
            for job in finished_this_tick:
                self._handle_job_completion(job)

    def print_results(self):
        self._log_average_inference_delay()
        self.training_log_file.close()
        self.usage_log_file.close()
        self.inference_delay_log_file.close()
        print("Simulation Complete. Results saved.")