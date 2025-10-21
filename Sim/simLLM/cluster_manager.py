# file: cluster_manager.py

from components import GPU, SHARABLE_GPU_MEM_PENALTY_GB

class ClusterManager:
    """Manages GPU pools and dynamic partial locking of sharable GPUs."""
    def __init__(self, num_training_gpus, num_inference_gpus):
        self.training_gpus = [GPU(f"T_{i}", 'training') for i in range(num_training_gpus)]
        self.inference_gpus = [GPU(f"I_{i}", 'inference') for i in range(num_inference_gpus)]
        
        num_sharable = num_inference_gpus // 2
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

    def get_locked_gpus(self):
        """Returns a list of all GPUs currently locked by the policy."""
        return [gpu for gpu in self.inference_gpus if gpu.is_reservable and not gpu.sharable]
    
    def lock_gpus(self, gpus_to_lock):
        """Locks a specific list of GPUs."""
        for gpu in gpus_to_lock:
            if gpu.is_reservable and gpu.sharable:
                gpu.sharable = False

    def unlock_gpus(self, gpus_to_unlock):
        """Unlocks a specific list of GPUs."""
        for gpu in gpus_to_unlock:
            if gpu.is_reservable and not gpu.sharable:
                gpu.sharable = True

    def find_gpu_for_stackable_inference(self, job):
        """Finds a single GPU that can fit a small inference job."""
        for gpu in sorted(self.inference_gpus, key=lambda g: g.is_idle()):
            # ** NEW: Only consider this GPU if it's NOT already running a borrowed training job. **
            if not gpu.get_running_training_jobs():
                if gpu.can_fit(job):
                    return gpu
        return None
    
    def find_gpu_for_llm_job(self):
        """Finds a single inference GPU with an available LLM slot."""
        for gpu in self.inference_gpus:
            if gpu.llm_slots_available > 0:
                return gpu
        return None
    
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

    def find_preemptible_job(self):
        for gpu in self.inference_gpus:
            if gpu.sharable:
                for job in gpu.get_running_training_jobs():
                    return (job, gpu)
        return (None, None)

    def release_resources_for_job(self, job):
        for gpu in job.assigned_gpus:
            gpu.release_task(job)