# file: cluster_manager.py

# --- (MODIFIED: Removed PREEMPTION_COOLDOWN import) ---
from components import GPU, LLM_MAX_CONCURRENCY

class ClusterManager:
    """Manages GPU pools and dynamic partial locking of sharable GPUs."""
    def __init__(self, num_training_gpus, num_inference_gpus):
        self.training_gpus = [GPU(f"T_{i}", 'training') for i in range(num_training_gpus)]
        self.inference_gpus = [GPU(f"I_{i}", 'inference') for i in range(num_inference_gpus)]
        
        print(f"ClusterManager initialized.")

    def get_inference_pool_utilization(self):
        total_used_util = sum((gpu.total_utilization - gpu.available_utilization) for gpu in self.inference_gpus)
        total_possible_util = sum(gpu.total_utilization for gpu in self.inference_gpus)
        return (total_used_util / total_possible_util) * 100 if total_possible_util > 0 else 0

    def find_gpu_for_stackable_inference(self, job):
        """Finds a single GPU that can fit a small inference job."""
        for gpu in sorted(self.inference_gpus, key=lambda g: g.is_idle()):
            # ** NEW: Only consider this GPU if it's NOT already running a borrowed training job. **
            # (Note: This check is now moot as training jobs can't run here, but it doesn't hurt)
            if not any(t['job'].job_type == 'training' for t in gpu.running_tasks.values()):
                if gpu.can_fit(job):
                    return gpu
        return None

    def find_resources_for_llm_batch(self, num_jobs_needed):
        """
        Finds available resources for an LLM batch with a multi-level priority.
        This function now returns a simple list of usable GPUs.
        """
        available_gpus = [] # This will be a list of GPU objects, not slots
        
        # --- Priority 1: Get all existing LLM servers ---
        active_server_gpus = [gpu for gpu in self.inference_gpus if gpu.is_llm_server]
        # active_server_gpus.sort(key=lambda gpu: gpu.llm_slots_available)
        
        available_gpus.extend(active_server_gpus)

        # --- Priority 2: Get all convertible idle GPUs ---
        convertible_gpus = [gpu for gpu in self.inference_gpus if not gpu.is_llm_server and gpu.is_idle()]

        # Just add the GPU *once*. The scheduler will handle filling it.
        available_gpus.extend(convertible_gpus)

        # Return the list of GPUs.
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

    # --- (REMOVED find_preemptible_job method) ---
    # def find_preemptible_job(self, current_time): ...

    def release_resources_for_job(self, job):
        for gpu in job.assigned_gpus:
            gpu.release_task(job)