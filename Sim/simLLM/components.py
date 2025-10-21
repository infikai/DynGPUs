# file: components.py

# --- Global Constants ---
GPU_MEMORY_GB = 32
GPU_UTILIZATION_PERCENT = 100
PREEMPTION_OVERHEAD = 3
RECLAMATION_OVERHEAD = 6 # 30 for Horovod
SHARABLE_GPU_MEM_PENALTY_GB = 1.5

# --- Policy Constants ---
# ** MODIFIED: Removed interval constants **
HIGH_UTIL_THRESHOLD = 80.0
LOW_UTIL_THRESHOLD = 50.0
PARTIAL_LOCK_FRACTION = 0.1

# NEW: LLM Inference Performance Model Constants
LLM_TTFT = 0.5  # Time To First Token (seconds)
LLM_TPOT = 0.05 # Time Per Output Token (seconds)
LLM_MAX_CONCURRENCY = 17 # Max concurrent requests per GPU

class SimulationClock:
    """A simple discrete-time simulation clock."""
    def __init__(self, tick_duration=1):
        self.current_time = 0
        self.tick_duration = tick_duration

    def tick(self):
        self.current_time += self.tick_duration

class GPU:
    def __init__(self, gpu_id, gpu_type):
        self.gpu_id = gpu_id
        self.gpu_type = gpu_type
        self.is_reservable = False
        self.sharable = False
        
        self.total_memory = GPU_MEMORY_GB
        self.total_utilization = GPU_UTILIZATION_PERCENT
        self.available_memory = self.total_memory
        self.available_utilization = self.total_utilization
        
        # NEW: Add an attribute to store utilization before LLM jobs take over
        self.utilization_before_llm = None

        self.llm_slots_total = 0
        self.llm_slots_available = 0
        if self.gpu_type == 'inference':
            self.llm_slots_total = LLM_MAX_CONCURRENCY
            self.llm_slots_available = LLM_MAX_CONCURRENCY
        
        self.running_tasks = {}

    def apply_memory_penalty(self, penalty_gb):
        self.total_memory -= penalty_gb
        self.available_memory -= penalty_gb

    def can_fit(self, job):
        return (job.memory_required <= self.available_memory and
                job.utilization_required <= self.available_utilization)

    def assign_task(self, job, mem_slice, util_slice):
        if mem_slice > self.available_memory or util_slice > self.available_utilization:
            print(f"\n‼️ WARNING: ASSIGNMENT FAILED ON GPU {self.gpu_id}")
            print(f"   GPU State : Available Mem={self.available_memory:.2f}, Available Util={self.available_utilization:.2f}")
            print(f"   Job Slice : Required Mem={mem_slice:.2f}, Required Util={util_slice:.2f}")
            print(f"   Full Job  : {job!r}\n")
            raise Exception(f"Resource slice for job {job.id}: mem:{mem_slice}, util:{util_slice} cannot fit on GPU {self.gpu_id}: {self.available_memory}: {self.available_utilization}.")
        self.available_memory -= mem_slice
        self.available_utilization -= util_slice
        self.running_tasks[job.id] = {'job': job, 'mem': mem_slice, 'util': util_slice}

    def assign_llm_task(self, job):
        if self.llm_slots_available <= 0:
            raise Exception(f"No LLM slots available on GPU {self.gpu_id}")

        # NEW: If this is the FIRST LLM job on this GPU, save and block utilization
        if self.llm_slots_available == self.llm_slots_total:
            self.utilization_before_llm = self.available_utilization
            self.available_utilization = 0

        self.llm_slots_available -= 1
        self.running_tasks[job.id] = {'job': job, 'type': 'llm'}

    def release_task(self, job):
        if job.id not in self.running_tasks: return
        
        task_info = self.running_tasks.pop(job.id)
        
        if task_info.get('type') == 'llm':
            self.llm_slots_available += 1
            
            # NEW: If this was the LAST LLM job, restore the original utilization
            if self.llm_slots_available == self.llm_slots_total and self.utilization_before_llm is not None:
                self.available_utilization = self.utilization_before_llm
                self.utilization_before_llm = None
        else:
            self.available_memory += task_info['mem']
            self.available_utilization += task_info['util']

    def is_idle(self):
        return not self.running_tasks

    def get_running_training_jobs(self):
        return [task['job'] for task in self.running_tasks.values() if task['job'].job_type == 'training']

class Job:
    def __init__(self, id, job_type, arrival_time, base_duration=0, 
                 memory_required=0, utilization_required=0, 
                 input_tokens=0, output_tokens=0):
        self.id = id
        self.job_type = job_type
        self.base_duration = base_duration
        self.arrival_time = arrival_time
        self.memory_required = max(memory_required, 1)
        self.utilization_required = max(utilization_required, 1)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

        # If it's an LLM job, calculate its duration now
        if self.job_type == 'llm_inference':
            self.base_duration = LLM_TTFT + (LLM_TPOT * self.output_tokens)
        
        self.remaining_work = base_duration
        self.assigned_gpus = []
        self.start_time = -1
        self.completion_time = -1
        self.turnaround_time = -1
        self.paused_until = -1
        self.gpus_needed = 1
        
    # ** NEW: Add __repr__ for detailed, developer-friendly printing **
    def __repr__(self):
        return (f"<Job id={self.id} type={self.job_type} "
                f"mem_req={self.memory_required:.2f} util_req={self.utilization_required:.2f} "
                f"gpus_needed={self.gpus_needed} duration={self.base_duration}>")

    def assign_resources(self, gpus, current_time):
        self.assigned_gpus = gpus
        self.start_time = current_time
        self._distribute_load()

    def _distribute_load(self):
        num_gpus = len(self.assigned_gpus)
        if num_gpus == 0: return

        mem_per_gpu = self.memory_required / num_gpus
        util_per_gpu = self.utilization_required / num_gpus
        
        for gpu in self.assigned_gpus:
            gpu.assign_task(self, mem_per_gpu, util_per_gpu)

    def preempt_and_pause(self, gpu_to_release, current_time):
        """Handles the logic of being preempted from one GPU."""
        
        # ** NEW DEBUG PRINTS: Show the state of the GPU during preemption **
        print(f"-> [DEBUG] PREEMPT START: Job {self.id} being preempted from GPU {gpu_to_release.gpu_id}.")
        print(f"   [DEBUG] GPU state BEFORE release: Mem={gpu_to_release.available_memory:.2f}, Util={gpu_to_release.available_utilization:.2f}, Tasks={gpu_to_release.running_tasks.keys()}")
        
        gpu_to_release.release_task(self)
        
        print(f"   [DEBUG] GPU state AFTER release: Mem={gpu_to_release.available_memory:.2f}, Util={gpu_to_release.available_utilization:.2f}, Tasks={gpu_to_release.running_tasks.keys()}")
        
        if gpu_to_release in self.assigned_gpus:
            self.assigned_gpus.remove(gpu_to_release)
        
        # Re-distribute load over remaining GPUs
        for gpu in self.assigned_gpus:
            gpu.release_task(self)

        if self.assigned_gpus:
            self._distribute_load()

        self.paused_until = current_time + PREEMPTION_OVERHEAD
        print(f"<- [DEBUG] PREEMPT END: Job {self.id} is now running on {len(self.assigned_gpus)} GPUs.")
        
    def reclaim_gpu(self, gpu_to_add, current_time):
        if gpu_to_add in self.assigned_gpus: return

        for gpu in self.assigned_gpus:
            gpu.release_task(self)

        self.assigned_gpus.append(gpu_to_add)
        self._distribute_load()
        
        self.paused_until = current_time + RECLAMATION_OVERHEAD
    
    def update_progress(self, time_delta, current_time):
        """Updates remaining work using a normalized speedup factor."""
        if not self.assigned_gpus or current_time < self.paused_until:
            return
        
        # ** Normalized speedup calculation **
        if self.gpus_needed > 0:
            speedup_factor = len(self.assigned_gpus) / self.gpus_needed
        else:
            speedup_factor = 1 # Fallback, should not happen

        self.remaining_work -= (time_delta * speedup_factor)
    
    def is_complete(self):
        return self.remaining_work <= 0

    def record_completion(self, current_time):
        self.completion_time = current_time
        self.turnaround_time = self.completion_time - self.arrival_time
