# file: components.py

import math

# ==========================================
# GLOBAL CONSTANTS & CONFIGURATION
# ==========================================

GPU_MEMORY_GB = 32
GPU_UTILIZATION_PERCENT = 100
LLM_MAX_CONCURRENCY = 16       # Max LLM slots per GPU

# LLM Performance Model Constants
LLM_BASE_TTFT = 2.5          # Base time to first token
LLM_TKN_PER_INPUT = 0.005    # Time per input token
LLM_TPOT = 0.1               # Time per output token

# DeepBoot Overhead Constants
OVERHEAD_COLD_START = 2.5       # Time to load model on FREE GPU
OVERHEAD_WARM_START = 0.01      # Time to start on PROTECT/RUN GPU
OVERHEAD_RECLAIM = 15.0         # Time to context switch a LOANED GPU
OVERHEAD_AFE_SYNC = 2.0         # Time for training job to sync after shrink

# DeepBoot Protection Time Formula
PROTECT_MIN = 30
PROTECT_MAX = 120
PROTECT_INTERVAL = 15


class SimulationClock:
    def __init__(self, tick_duration=1):
        self.current_time = 0
        self.tick_duration = tick_duration

    def tick(self):
        self.current_time += self.tick_duration


class GPU:
    def __init__(self, gpu_id, gpu_type):
        self.gpu_id = gpu_id
        self.gpu_type = gpu_type      # 'training' or 'inference'
        
        self.state = 'FREE' 
        self.protect_time_remaining = 0
        self.usage_count = 0  
        
        self.total_memory = GPU_MEMORY_GB
        self.total_utilization = GPU_UTILIZATION_PERCENT
        
        # Initialize available resources
        self.available_memory = self.total_memory
        self.available_utilization = self.total_utilization
        
        self.running_tasks = {}

    def update_lifecycle(self, tick_duration):
        """Updates the PROTECT timer. Transitions PROTECT -> FREE on timeout."""
        if self.state == 'PROTECT':
            self.protect_time_remaining -= tick_duration
            if self.protect_time_remaining <= 0:
                self.state = 'FREE'
                self.usage_count = 0 
    
    def is_idle(self):
        """Helper to check if GPU is free for tasks."""
        return self.state == 'FREE' and not self.running_tasks
    
    @property
    def is_llm_server(self):
        return self.state in ['RUN', 'PROTECT']
    
    @property
    def llm_slots_available(self):
        if self.state == 'RUN':
            return LLM_MAX_CONCURRENCY - len(self.running_tasks)
        elif self.state == 'PROTECT':
            return LLM_MAX_CONCURRENCY
        return 0

    def convert_to_llm_server(self):
        """Transitions FREE -> RUN/PROTECT context."""
        if self.state == 'FREE':
            self.state = 'RUN' 
            return True
        return False
        
    def revert_from_llm_server(self):
        """Transitions PROTECT -> FREE."""
        if self.state == 'PROTECT':
            self.state = 'FREE'
            self.usage_count = 0
            return True
        return False

    def calculate_protection_time(self):
        """DeepBoot Formula: t_p = t_p,min + g.cnt * interval"""
        raw_time = PROTECT_MIN + (self.usage_count * PROTECT_INTERVAL)
        return min(PROTECT_MAX, raw_time)

    def assign_task(self, job):
        """Assigns a task. For Inference, state becomes RUN. For Training, TRAIN."""
        self.running_tasks[job.id] = {'job': job}
        
        # DeepBoot State Transitions on Assignment
        if job.job_type in ['inference', 'llm_inference']:
            self.state = 'RUN'
        else:
            self.state = 'TRAIN'
            self.available_memory -= job.memory_required

    def assign_llm_task(self, job):
        self.assign_task(job)

    def release_task(self, job):
        if job.id in self.running_tasks:
            del self.running_tasks[job.id]
            
            if job.job_type == 'training':
                self.available_memory += job.memory_required
    
    # Support for stackable inference checks
    def can_fit(self, job):
        return (job.memory_required <= self.available_memory and
                job.utilization_required <= self.available_utilization)

    def __repr__(self):
        return f"<GPU {self.gpu_id} ({self.gpu_type}) State={self.state} Tasks={len(self.running_tasks)}>"


class Job:
    def __init__(self, id, job_type, arrival_time, base_duration=0, 
                 memory_required=1, utilization_required=1,
                 input_tokens=0, output_tokens=0):
        self.id = id
        self.job_type = job_type  # 'training', 'inference', 'llm_inference'
        self.arrival_time = arrival_time
        
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        
        # Calculate duration for LLM jobs if not provided
        if self.job_type == 'llm_inference' and base_duration == 0:
            self.base_duration = (LLM_BASE_TTFT + 
                                  (LLM_TKN_PER_INPUT * input_tokens) + 
                                  (LLM_TPOT * output_tokens))
        else:
            self.base_duration = base_duration

        self.remaining_work = self.base_duration
        self.memory_required = max(memory_required, 1)
        self.utilization_required = max(utilization_required, 1)
        
        self.assigned_gpus = []
        self.start_time = -1
        self.completion_time = -1
        self.turnaround_time = -1
        self.gpus_needed = 1  
        
        self.overhead_remaining = 0.0

    def assign_resources(self, gpus, current_time):
        self.assigned_gpus = gpus
        if self.start_time == -1:
            self.start_time = current_time

    def calculate_speedup(self, num_gpus):
        if num_gpus <= 0: return 0.0
        # Linear scaling as requested
        return num_gpus ** 1.0

    def update_progress(self, time_delta, current_time):
        if not self.assigned_gpus:
            return

        # 1. Consume Overhead
        if self.overhead_remaining > 0:
            deduct = min(self.overhead_remaining, time_delta)
            self.overhead_remaining -= deduct
            time_delta -= deduct 
        
        # 2. Consume Real Work
        if time_delta > 0 and self.overhead_remaining <= 0:
            if self.job_type == 'training':
                speedup = self.calculate_speedup(len(self.assigned_gpus))
                self.remaining_work -= (time_delta * speedup)
            else:
                self.remaining_work -= time_delta

    def is_complete(self):
        return self.remaining_work <= 0

    def record_completion(self, current_time):
        self.completion_time = current_time
        self.turnaround_time = self.completion_time - self.arrival_time
    
    def __repr__(self):
        return f"[{self.id} | {self.job_type} | Rem:{self.remaining_work:.1f}s]"