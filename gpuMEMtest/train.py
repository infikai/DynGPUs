import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import horovod.torch as hvd
import time
import os
import socket
import logging
import sys
# Assuming MyElasticSampler is in the same directory or path
try:
    from sampler import MyElasticSampler
except ImportError:
    # Fallback if sampler file isn't present for this run
    from torch.utils.data.distributed import DistributedSampler as MyElasticSampler 

import argparse

# --- CONFIGURATION ---
TARGET_BATCHES = 5  # Stop after this many batches
# ---------------------

def monitor_gpu_memory(stage, rank):
    """
    Prints the GPU memory breakdown using the user's provided logic.
    Only prints on the local rank 0 to avoid console spam.
    """
    if rank != 0: 
        return

    # Synchronize to ensure all ops are done before measuring
    torch.cuda.synchronize()
    
    # Get raw data
    free_mem, total_mem = torch.cuda.mem_get_info(hvd.local_rank())
    reserved = torch.cuda.memory_reserved(hvd.local_rank())
    allocated = torch.cuda.memory_allocated(hvd.local_rank())
    
    used_mem_driver = total_mem - free_mem
    context_overhead = used_mem_driver - reserved
    fragmentation = reserved - allocated
    
    print(f"\n[{stage}] GPU Memory Snapshot:")
    print(f"  Total GPU Memory:       {total_mem / 1024**2:.2f} MB")
    print(f"  Driver/Context Overhead: {context_overhead / 1024**2:.2f} MB")
    print(f"  PyTorch Reserved:       {reserved / 1024**2:.2f} MB")
    print(f"    - Actual Tensors:     {allocated / 1024**2:.2f} MB")
    print(f"    - Fragmentation:      {fragmentation / 1024**2:.2f} MB\n")

class TrainingState:
    def __init__(self):
        self.epoch = 0
        self.batch_idx = 0
        self.processed_num = 0
        self.seed = 0

def main():
    parser = argparse.ArgumentParser(description='PyTorch Memory Profiling')
    parser.add_argument('--model', default='resnet50', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    args = parser.parse_args()

    hvd.init(process_sets="dynamic")
    
    # Set device immediately
    torch.cuda.set_device(hvd.local_rank())

    # STAGE 1: Baseline (Context Created)
    monitor_gpu_memory("1. Baseline (Context Created)", hvd.rank())

    if hvd.rank() == 0:
        logging.basicConfig(filename='worker_adjustments.log', level=logging.INFO)

    print(f"==> Using model: {args.model} | Batch Size: {args.batch_size}")
    
    # Create Model
    if args.model == 'resnet50':
        model = models.resnet50()
    elif args.model == 'vit_l_32':
        model = models.vit_l_32(weights=None)
    
    # Move to CUDA
    model.cuda()
    
    # STAGE 2: Model Loaded
    monitor_gpu_memory("2. Model Loaded to GPU", hvd.rank())

    base_optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Use dummy data if CIFAR not found, or standard CIFAR
    try:
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                            transform=transforms.Compose([transforms.ToTensor(), 
                                            transforms.Normalize((0.5,), (0.5,))]))
    except:
        print("Dataset not found, ensure ./data exists")
        return

    # Use DistributedSampler if MyElasticSampler fails (for standalone testing)
    try:
        sampler = MyElasticSampler(train_dataset)
    except:
         sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

    state = TrainingState()
    # Force current rank to be active for this test
    current_active_ranks = list(range(hvd.size())) 
    active_set = None 
    hvd_optimizer = hvd.DistributedOptimizer(base_optimizer, named_parameters=model.named_parameters())

    local_rank = current_active_ranks.index(hvd.rank())
    
    # Loader setup
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2, sampler=sampler)
    data_iterator = iter(loader)
    
    print(f"Starting training for {TARGET_BATCHES} batches...")
    
    batches_run = 0

    # Training Loop
    while batches_run < TARGET_BATCHES:
        try:
            images, target = next(data_iterator)
            images, target = images.cuda(), target.cuda()
            
            hvd_optimizer.zero_grad()
            output = model(images)
            loss = F.cross_entropy(output, target)
            loss.backward()
            hvd_optimizer.step()
            
            batches_run += 1
            print(f"Rank {hvd.rank()} - Batch {batches_run}/{TARGET_BATCHES} complete.")

            if batches_run == 1:
                # STAGE 3: Peak Training Memory (Activations + Gradients present)
                monitor_gpu_memory("3. Training (Activations + Grads)", hvd.rank())

        except StopIteration:
            break

    print("Target batches reached. Initiating offload...")

    # STAGE 4: Offload to CPU (Simulating Elastic Scale-Down)
    # 1. Zero grads to clear some buffers
    hvd_optimizer.zero_grad()
    
    # 2. Move Optimizer State
    move_optimizer_state(base_optimizer, 'cpu')
    
    # 3. Move Model
    model.cpu()
    
    # 4. Delete CUDA tensors (images/targets)
    # del images
    # del target
    # del output
    # del loss
    
    # 5. Empty Cache
    torch.cuda.empty_cache()
    
    # STAGE 5: Post-Cleanup
    monitor_gpu_memory("4. Post-Cleanup (CPU Offload)", hvd.rank())

    print("Done.")

def move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

if __name__ == "__main__":
    main()