# file: cluster_manager.py

from components import GPU, SHARABLE_GPU_MEM_PENALTY_GB, LLM_MAX_CONCURRENCY, PREEMPTION_COOLDOWN

class ClusterManager:
    """Manages GPU pools and dynamic partial locking of sharable GPUs."""
    def __init__(self, num_training_gpus, num_inference_gpus):
        self.training_gpus = [GPU(f"T_{i}", 'training') for i in range(num_training_gpus)]
        self.inference_gpus = [GPU(f"I_{i}", 'inference') for i in range(num_inference_gpus)]
        
        num_sharable = num_inference_gpus // 5 * 4
        for i in range(num_sharable):
            gpu = self.inference_gpus[i]
            gpu.is_reservable = True
            gpu.sharable = True
            gpu.apply_memory_penalty(SHARABLE_GPU_MEM_PENALTY_GB)
        
        self.num_reservable_gpus = num_sharable
        print(f"ClusterManager initialized. {self.num_reservable_gpus} inference GPUs are initially 'sharable'.")

    def get_inference_pool_utilization(self):
        total_used_util = sum((gpu.total_utilization - gpu.available_utilization) for gpu in self.inference_gpus)
        total_possible_util = sum(gpu.total_utilization for gpu in self.inference_gpus)
        return (total_used_util / total_possible_util) * 100 if total_possible_util > 0 else 0

    def get_idle_sharable_gpus(self):
        """Returns a list of all currently idle and sharable GPUs."""
        return [gpu for gpu in self.inference_gpus if gpu.is_idle() and gpu.sharable]

    # def get_locked_gpus(self):
    #     """Returns a list of all GPUs currently locked by the policy."""
    #     return [gpu for gpu in self.inference_gpus if gpu.is_reservable and not gpu.sharable]
    
    # def lock_gpus(self, gpus_to_lock):
    #     """Locks a specific list of GPUs."""
    #     for gpu in gpus_to_lock:
    #         if gpu.is_reservable and gpu.sharable:
    #             gpu.sharable = False

    # def unlock_gpus(self, gpus_to_unlock):
    #     """Unlocks a specific list of GPUs."""
    #     for gpu in gpus_to_unlock:
    #         if gpu.is_reservable and not gpu.sharable:
    #             gpu.sharable = True

    def find_gpu_for_stackable_inference(self, job):
        """Finds a single GPU that can fit a small inference job."""
        for gpu in sorted(self.inference_gpus, key=lambda g: g.is_idle()):
            # ** NEW: Only consider this GPU if it's NOT already running a borrowed training job. **
            if not gpu.get_running_training_jobs():
                if gpu.can_fit(job):
                    return gpu
        return None

    def find_resources_for_llm_batch(self, num_jobs_needed):
        """
        Finds available resources for an LLM batch with a multi-level priority.
        This function now returns a simple list of usable GPUs.
        """
        available_gpus = [] # This will be a list of GPU objects, not slots
        
        # --- Priority 1 & 2: Get all existing LLM servers ---
        active_server_gpus = [gpu for gpu in self.inference_gpus if gpu.is_llm_server]
        active_server_gpus.sort(key=lambda gpu: (gpu.sharable, gpu.llm_slots_available))
        
        available_gpus.extend(active_server_gpus)

        # # --- Priority 3 & 4: Get all convertible idle GPUs ---
        # convertible_gpus = [gpu for gpu in self.inference_gpus if not gpu.is_llm_server and gpu.is_idle()]
        # convertible_gpus.sort(key=lambda gpu: gpu.sharable)

        # # Just add the GPU *once*. The scheduler will handle filling it.
        # available_gpus.extend(convertible_gpus)

        # Return the list of GPUs. The adaptive policy will handle preemption.
        return available_gpus, []
    
    def find_gpu_for_llm_job(self):
        """
        Finds a single GPU for an LLM job using the new dynamic server logic.
        
        Priority:
        1. Find an existing LLM server with available slots.
        2. Find an idle regular inference GPU that can be converted.
        """
        # 1. Prioritize existing LLM servers with free slots
        for gpu in self.inference_gpus:
            if gpu.is_llm_server and gpu.llm_slots_available > 0:
                return gpu
        
        # 2. If none found, find an idle regular GPU to convert
        for gpu in self.inference_gpus:
            if not gpu.is_llm_server and gpu.is_idle():
                return gpu
                
        return None # Return None if no suitable GPU is found
    
    def find_idle_gpus_in_inference_pool(self, count):
        """Finds a specific number of idle GPUs from the entire inference pool."""
        idle_gpus = [gpu for gpu in self.inference_gpus if gpu.is_idle()]
        if len(idle_gpus) >= count:
            return idle_gpus[:count]
        return []

    def find_idle_gpus_for_training(self, count):
        idle_gpus = [gpu for gpu in self.training_gpus if gpu.is_idle()]
        return idle_gpus[:count] if len(idle_gpus) >= count else []
    
    def find_idle_borrowable_gpus(self, count):
        borrowable_gpus = [gpu for gpu in self.inference_gpus if gpu.is_idle() and gpu.sharable]
        return borrowable_gpus[:count]

    def find_preemptible_job(self, current_time):
        """
        Finds the best training job to preempt based on a Least Recently Preempted policy.
        """
        potential_victims = []
        
        # 1. Find all possible victims running on sharable, inference GPUs.
        for gpu in self.inference_gpus:
            if gpu.sharable:
                for job in gpu.get_running_training_jobs():
                    # NEW: Only consider this job a victim if it's not in its cooldown period.
                    if current_time > job.last_preemption_time + PREEMPTION_COOLDOWN:
                        potential_victims.append((job, gpu))
                    
        # 2. If no victims exist, return None.
        if not potential_victims:
            return (None, None)
            
        # 3. Sort the list of victims by their last preemption time (ascending).
        #    Jobs that have never been preempted (time = -1) will automatically be first.
        potential_victims.sort(key=lambda item: item[0].last_preemption_time)
        
        # 4. Return the best victim (the first one in the sorted list).
        return potential_victims[0]

    def release_resources_for_job(self, job):
        for gpu in job.assigned_gpus:
            gpu.release_task(job)