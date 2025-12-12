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
from sampler import MyElasticSampler
import argparse
import logging
import gc

# Hyperparameters
EPOCHS = 90

# --- MEMORY MONITORING HELPER ---
def monitor_gpu_memory(stage, rank):
    """
    Prints the GPU memory breakdown.
    Only prints on rank 0 (or specific ranks) to avoid console spam.
    """
    # You can adjust this condition if you want to see logs from other ranks

    # Synchronize to ensure all ops are done before measuring
    torch.cuda.synchronize()
    
    # local_rank = hvd.local_rank()
    if hvd.local_rank() == 0:
        local_rank = 1
    elif hvd.local_rank() == 1:
        local_rank = 2
    elif hvd.local_rank() == 2:
        local_rank = 0
    else:
        local_rank = hvd.local_rank()
    
    # Get raw data
    try:
        free_mem, total_mem = torch.cuda.mem_get_info(local_rank)
        reserved = torch.cuda.memory_reserved(local_rank)
        allocated = torch.cuda.memory_allocated(local_rank)
        
        used_mem_driver = total_mem - free_mem
        context_overhead = used_mem_driver - reserved
        fragmentation = reserved - allocated
        
        print(f"\n[{stage}] GPU Memory Snapshot (Rank {rank}):")
        print(f"  Total GPU Memory:       {total_mem / 1024**2:.2f} MB")
        print(f"  Driver/Context Overhead: {context_overhead / 1024**2:.2f} MB")
        print(f"  PyTorch Reserved:       {reserved / 1024**2:.2f} MB")
        print(f"    - Actual Tensors:     {allocated / 1024**2:.2f} MB")
        print(f"    - Fragmentation:      {fragmentation / 1024**2:.2f} MB\n")
    except Exception as e:
        print(f"Memory monitor failed: {e}")
# -------------------------------

class TrainingState:
    def __init__(self):
        self.epoch = 0
        self.batch_idx = 0
        self.processed_num = 0
        self.seed = 0

def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training with Dynamic GPUs and CLI args')
    parser.add_argument('--model', default='resnet50', type=str,
                        help='model to train (e.g., resnet50, vit_l_32)',
                        choices=['resnet50', 'vit_l_32'])
    parser.add_argument('--batch_size', default=128, type=int,
                        help='input batch size for training')
    args = parser.parse_args()

    hvd.init(process_sets="dynamic")

    if hvd.rank() == 0:
        logging.basicConfig(filename='worker_adjustments.log',
                            level=logging.INFO,
                            format='%(asctime)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        logging.info("Starting training run.")

    if hvd.rank() == 1:
        logging.basicConfig(filename='throughput.log',
                            level=logging.INFO,
                            format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

    hostname = socket.gethostname()
    parts = hostname.split('.')
    nodename = parts[0]
    print(f'{hostname} binded to horovod rank{hvd.rank()}.')

    # Set device
    if hvd.local_rank() == 0:
        device_id = 1
    elif hvd.local_rank() == 1:
        device_id = 2
    elif hvd.local_rank() == 2:
        device_id = 0
    else:
        device_id = hvd.local_rank()
    
    torch.cuda.set_device(device_id)

    device = torch.device(f'cuda:{device_id}')

    # [MEMORY] Stage 1: Context Created
    monitor_gpu_memory("1. Baseline (Context Created)", hvd.rank())

    ST_model = time.time()
    print(f"==> Using model: {args.model} | Batch Size: {args.batch_size}")
    if args.model == 'resnet50':
        model = models.resnet50().cuda()
    elif args.model == 'vit_l_32':
        model = models.vit_l_32(weights=None).cuda()
    else:
        raise ValueError(f"Unsupported model specified: {args.model}")

    # [MEMORY] Stage 2: Model Loaded (Initial)
    monitor_gpu_memory("2. Model Loaded to GPU", hvd.rank())

    base_optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train_dataset = datasets.CIFAR10(root='./data', 
                                        train=True, 
                                        download=True,
                                        transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    sampler = MyElasticSampler(train_dataset)

    state = TrainingState()
    current_active_ranks = []
    process_set_cache = {}
    hvd_optimizer = None
    fisrt_batch = True
    
    last_log_time = time.time()
    processed = 0

    # Train
    while state.epoch < EPOCHS:
        config_changed = True
        epoch_finished = 0

        while True:
            if config_changed:
                if hvd.rank() == 1:
                        logging.info(f'Throughput: 0 images/second.')
                print("Config changing")
                ST_config = time.time()
                fisrt_batch = True
                sync = True

                monitor_gpu_memory("2.1 before bcast", hvd.rank())
                if hvd.rank() == 0:
                    active_ranks = read_active_ranks_from_file()
                else:
                    active_ranks = None
                active_ranks = hvd.broadcast_object(active_ranks, root_rank=0, name="ranks_bcast")
                monitor_gpu_memory("2.1 after bcast", hvd.rank())

                old_active_ranks = current_active_ranks
                print(f'Old ranks: {old_active_ranks}')
                current_active_ranks = active_ranks
                print(f'New ranks: {current_active_ranks}')
                is_full_world = (len(current_active_ranks) == hvd.size())
                unique_in_new = list(set(current_active_ranks) - set(old_active_ranks))
                print(f'Unique in new ranks: {unique_in_new}')
                if len(unique_in_new) == 0:
                    sync = False

                if hvd.rank() not in old_active_ranks and hvd.rank() in current_active_ranks:
                    ST_moveOP = time.time()
                    print(hvd.rank())
                    move_optimizer_state(base_optimizer, 'cuda')
                    print(f'Move OP to cuda Cost: {time.time() - ST_moveOP}s')

                if is_full_world:
                    active_set = None
                else:
                    ranks_tuple = tuple(sorted(current_active_ranks))
                    needs_creation_local = torch.tensor(1 if ranks_tuple not in process_set_cache else 0)
                    needs_creation_sum = hvd.allreduce(needs_creation_local, name=f"creation_lock_{ranks_tuple}")
                    if needs_creation_sum.item() > 0:
                        process_set_cache[ranks_tuple] = hvd.add_process_set(current_active_ranks)
                    active_set = process_set_cache[ranks_tuple]

                if hvd.rank() in current_active_ranks:
                    model.cuda() # Ensure model is on CUDA
                    root_rank_for_sync = 0
                    if sync:
                        ST_bcast = time.time()
                        if is_full_world:
                            print('=== Full world case ===')
                            hvd.broadcast_parameters(model.state_dict(), root_rank=root_rank_for_sync)
                            hvd.broadcast_optimizer_state(base_optimizer, root_rank=root_rank_for_sync)
                            state = hvd.broadcast_object(state, root_rank=root_rank_for_sync, name="BcastState")
                            print(f'Whole BCAST cost: {time.time() - ST_bcast}s')
                        else:
                            print('=== Partial world case ===')
                            hvd.broadcast_parameters(model.state_dict(), root_rank=root_rank_for_sync, process_set=active_set)
                            hvd.broadcast_optimizer_state(base_optimizer, root_rank=root_rank_for_sync, process_set=active_set)
                            state = hvd.broadcast_object(state, root_rank=root_rank_for_sync, process_set=active_set, name="BcastState")
                            print(f'Whole BCAST cost: {time.time() - ST_bcast}s')
                    print('==='*5)

                    if hvd_optimizer != None:
                        hvd_optimizer._unregister_hooks()
                    if active_set is None:
                        hvd_optimizer = hvd.DistributedOptimizer(base_optimizer, named_parameters=model.named_parameters())
                    else:
                        hvd_optimizer = hvd.DistributedOptimizer(base_optimizer, named_parameters=model.named_parameters(), process_set=active_set)

                    local_rank = current_active_ranks.index(hvd.rank())
                    ST_data = time.time()
                    sampler.set_epoch(state.epoch, state.processed_num, num_replicas=len(current_active_ranks), rank=local_rank)
                    loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, sampler=sampler)
                    print(f'Data Cost: {time.time() - ST_data}s')
                    data_iterator = iter(loader)

                config_change_duration = time.time() - ST_config
                print(f'Config Change Cost: {config_change_duration}s')
                
                if hvd.rank() == 0:
                    logging.info(f"Configuration change took {config_change_duration:.4f} seconds. Sync required: {sync}")

                config_changed = False
                if hvd.rank() == 1:
                        logging.info(f'Throughput: 0 images/second.')

            if hvd.rank() == 0:
                new_ranks = read_active_ranks_from_file()
            else:
                new_ranks = None
            new_ranks = hvd.broadcast_object(new_ranks, root_rank=0, name="ranks_check_bcast")
            if new_ranks != current_active_ranks:
                if hvd.rank() == 0:
                    logging.info(f"Adjusting workers. Previous: {sorted(current_active_ranks)}, New: {sorted(new_ranks)}")
                config_changed = True
                break

            if hvd.rank() in current_active_ranks:
                try:
                    ST_batch = time.time()
                    images, target = next(data_iterator)
                    images, target = images.cuda(), target.cuda()
                    if fisrt_batch:
                         monitor_gpu_memory("3.0 Training (Activations + Grads)", hvd.rank())
                    hvd_optimizer.zero_grad()
                    output = model(images)
                    if fisrt_batch:
                         monitor_gpu_memory("3.1 Training (Activations + Grads)", hvd.rank())
                    loss = F.cross_entropy(output, target)
                    if fisrt_batch:
                         monitor_gpu_memory("3.2 Training (Activations + Grads)", hvd.rank())
                    loss.backward()
                    if fisrt_batch:
                         monitor_gpu_memory("3.3 Training (Activations + Grads)", hvd.rank())
                    hvd_optimizer.step()
                    
                    # [MEMORY] Stage 3: Training (First batch of new config)
                    if fisrt_batch:
                         monitor_gpu_memory("3.4 Training (Activations + Grads)", hvd.rank())

                    print(f'One Batch Cost: {time.time() - ST_batch}s')
                    sampler.record_batch(state.batch_idx, args.batch_size)
                    images = args.batch_size * len(current_active_ranks)
                    processed += images
                    state.batch_idx += 1
                    state.processed_num = sampler.get_processed_num()
                    if hvd.rank() == current_active_ranks[0]:
                        print(f"Epoch: {state.epoch} | Batch: {state.batch_idx-1} | Loss: {loss.item():.4f}")
                        
                    if hvd.rank() == 0 and fisrt_batch == True:
                            logging.info(f"First Batch Cost: {time.time() - ST_batch}s")
                    fisrt_batch = False
                    time.sleep(2)
                    END_batch = time.time()

                    current_time = time.time()
                    if hvd.rank() == 0 and (current_time - last_log_time) >= 3:
                        logging.info(f"Progress - Epoch: {state.epoch}, Processed: {processed}")
                        last_log_time = current_time
                    throughput = images / (END_batch - ST_batch)
                    if hvd.rank() == 1:
                        logging.info(f'Throughput: {throughput} images/second.')

                except StopIteration:
                    epoch_finished = 1
            
            # INACTIVE/OFFLOAD BLOCK
            if hvd.rank() not in current_active_ranks:
                if hvd_optimizer is not None:
                    # 1. Clear Grads
                    hvd_optimizer.zero_grad()
                    # 2. Unregister hooks (Crucial for Horovod to release model refs)
                    try:
                        hvd_optimizer._unregister_hooks()
                    except:
                        pass
                    
                    # 3. Destroy the object explicitly
                    del hvd_optimizer
                    hvd_optimizer = None
                
                move_optimizer_state(base_optimizer, 'cpu')
                model.cpu()

                try:
                    del images
                    del target
                    del output
                    del loss
                except NameError:
                    pass
                gc.collect()
                
                # Clear CUDA cache to see true drop in usage
                torch.cuda.empty_cache()

                # [MEMORY] Stage 4: Inactive
                monitor_gpu_memory("4. Inactive (CPU Offload)", hvd.rank())

                # time.sleep(1) # Original sleep commented out in provided code
            
            finished_tensor = torch.tensor(epoch_finished)
            finished_tensor = hvd.broadcast_object(finished_tensor, root_rank=0, name="epoch_end_bcast")
            if finished_tensor.item() == 1:
                break

        if not config_changed:
            state.epoch += 1
            state.batch_idx = 0
            state.processed_num = 0
            config_changed = True

def read_active_ranks_from_file(filepath='/home/pacs/Kevin/DynGPUs/gpuMEMtest/active_workers.txt'):
    try:
        if not os.path.exists(filepath):
            time.sleep(1)
            if not os.path.exists(filepath): return list(range(hvd.size()))
        with open(filepath, 'r') as f:
            content = f.read().strip()
        if not content: return []
        return sorted([int(r) for r in content.split(',')])
    except Exception as e:
        print(f"Error reading file: {e}. Defaulting to all ranks.")
        return list(range(hvd.size()))

def allreduce_gradients_manual(model, process_set, name):
    params_with_grad = [p for p in model.parameters() if p.grad is not None]
    if not params_with_grad:
        return
    flat_grads = torch.cat([p.grad.view(-1) for p in params_with_grad])
    if process_set is None:
        hvd.allreduce_(flat_grads, average=True, name=name)
    else:
        hvd.allreduce_(flat_grads, average=True, process_set=process_set, name=name)
    offset = 0
    for p in params_with_grad:
        numel = p.grad.numel()
        p.grad.copy_(flat_grads[offset:offset + numel].view_as(p.grad))
        offset += numel

def move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

if __name__ == "__main__":
    main()