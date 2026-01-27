# file: cluster_manager.py

from components import GPU, LLM_MAX_CONCURRENCY

class ClusterManager:
    def __init__(self, num_training_gpus, num_inference_gpus):
        self.training_gpus = [GPU(f"T_{i}", 'training') for i in range(num_training_gpus)]
        self.inference_gpus = [GPU(f"I_{i}", 'inference') for i in range(num_inference_gpus)]
        print(f"ClusterManager initialized: {num_training_gpus} Training GPUs, {num_inference_gpus} Inference GPUs.")

    # --- ATS-I: Inference Selection Helpers ---

    def find_warm_inference_gpu(self):
        """
        Priority 1: Find a GPU ready for LLM inference.
        Sub-priority A: A GPU in 'RUN' state with available slots (Stacking).
        Sub-priority B: A GPU in 'PROTECT' state (Empty but warm).
        """
        # A. Check for Stacking (Best Utilization)
        for gpu in self.inference_gpus:
            if gpu.state == 'RUN' and len(gpu.running_tasks) < LLM_MAX_CONCURRENCY:
                return gpu
        
        # B. Check for Warm Idle (Protection Pool)
        candidates = [g for g in self.inference_gpus if g.state == 'PROTECT']
        if candidates:
            # Sort by usage_count desc (g.cnt)
            candidates.sort(key=lambda x: x.usage_count, reverse=True)
            return candidates[0]
            
        return None

    def find_free_inference_gpu(self):
        """Priority 2: Find FREE GPU (Cold Start)."""
        for gpu in self.inference_gpus:
            if gpu.state == 'FREE':
                return gpu
        return None

    def find_reclaim_target(self):
        """Priority 3: Find a LOANED GPU (TRAIN state) to preempt."""
        best_gpu = None
        min_loss = float('inf')

        # Only look at GPUs currently loaned to training
        loaned_gpus = [g for g in self.inference_gpus if g.state == 'TRAIN']

        for gpu in loaned_gpus:
            if not gpu.running_tasks: continue
            
            job = list(gpu.running_tasks.values())[0]['job']
            current_gpus = len(job.assigned_gpus)
            
            # Don't kill the last GPU of a training job if we can avoid it
            if current_gpus <= 1:
                loss = float('inf')
            else:
                curr_speed = job.calculate_speedup(current_gpus)
                next_speed = job.calculate_speedup(current_gpus - 1)
                loss = curr_speed - next_speed
            
            if loss < min_loss:
                min_loss = loss
                best_gpu = gpu
                
        return best_gpu

    # --- Training Selection Helpers ---

    def find_idle_training_gpus(self, count):
        """Finds idle dedicated training GPUs."""
        idle = []
        for gpu in self.training_gpus:
            if not gpu.running_tasks:
                idle.append(gpu)
                if len(idle) == count: break
        return idle

    def find_loanable_inference_gpus(self, count):
        """Finds 'count' inference GPUs that are strictly FREE."""
        loanable = []
        for gpu in self.inference_gpus:
            if gpu.state == 'FREE':
                loanable.append(gpu)
                if len(loanable) == count: break
        return loanable