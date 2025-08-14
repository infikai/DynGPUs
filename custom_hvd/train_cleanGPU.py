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
from torch.utils.data.distributed import DistributedSampler
import socket
BATCH_SIZE = 64

class TrainingState:
    def __init__(self):
        self.epoch = 0
        self.batch_idx = 0

def main():
    hvd.init(process_sets="dynamic")
    hostname = socket.gethostname()
    print(f'Node: {hostname} binded rank is {hvd.rank()}')
    torch.cuda.set_device(hvd.local_rank())
    # model = models.resnet50().cuda()
    model = models.vit_l_32(weights=None).cuda()
    # Use a standard PyTorch optimizer. We will manage the gradient reduction manually.
    base_optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # criterion = nn.CrossEntropyLoss().cuda()
    train_dataset = datasets.ImageFolder(
        os.path.join('/mydata/Data/imagenet', 'train'),
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))
    state = TrainingState()
    current_active_ranks = []
    process_set_cache = {}

    while state.epoch < 100:
        config_changed = True
        
        while True:
            if config_changed:
                ST_config = time.time()
                # This logic for determining the active set remains the same
                if hvd.rank() == 0:
                    active_ranks = read_active_ranks_from_file()
                else:
                    active_ranks = None
                active_ranks = hvd.broadcast_object(active_ranks, root_rank=0, name="ranks_bcast")

                current_active_ranks = active_ranks
                is_full_world = (len(current_active_ranks) == hvd.size())

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
                    model.cuda()
                    base_optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
                    # State synchronization logic remains the same
                    root_rank_for_sync = current_active_ranks[0] if not is_full_world else 0
                    ST_bcast = time.time()
                    if is_full_world:
                        print('full world case')
                        # Case 1: All workers active. Use the efficient built-in functions.
                        hvd.broadcast_parameters(model.state_dict(), root_rank=root_rank_for_sync)
                        hvd.broadcast_optimizer_state(base_optimizer, root_rank=root_rank_for_sync)
                        state = hvd.broadcast_object(state, root_rank=root_rank_for_sync, name="BcastState")
                        print(f'fBCAST cost: {time.time() - ST_bcast}s')
                    else:
                        print('partial world case')
                        # Case 2: A subset is active. Use the manual object broadcast.
                        hvd.broadcast_parameters(model.state_dict(), root_rank=root_rank_for_sync, process_set=active_set)
                        if hvd.rank() == root_rank_for_sync:
                            # model_state = model.state_dict()
                            opt_state = base_optimizer.state_dict()
                        else:
                            opt_state = None

                        # bcast_model_state = hvd.broadcast_object(model_state, root_rank=root_rank_for_sync, process_set=active_set, name="BcastModel")
                        bcast_opt_state = hvd.broadcast_object(opt_state, root_rank=root_rank_for_sync, process_set=active_set, name="BcastOpt")
                        
                        if hvd.rank() != root_rank_for_sync:
                            # model.load_state_dict(bcast_model_state)
                            base_optimizer.load_state_dict(bcast_opt_state)

                        state = hvd.broadcast_object(state, root_rank=root_rank_for_sync, process_set=active_set, name="BcastState")
                        print(f'pBCAST cost: {time.time() - ST_bcast}s')

                    # Data loader setup remains the same
                    local_rank = current_active_ranks.index(hvd.rank())
                    sampler = DistributedSampler(train_dataset, num_replicas=len(current_active_ranks), rank=local_rank)
                    sampler.set_epoch(state.epoch)
                    loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, sampler=sampler)
                    data_iterator = iter(loader)
                    ST_fast_forward = time.time()
                    for _ in range(state.batch_idx):
                        next(data_iterator)
                    print(f'Fast froward cost: {time.time() - ST_fast_forward}s')
                print(f'Config Change Cost: {time.time() - ST_config}s')
                config_changed = False

            if hvd.rank() == 0:
                new_ranks = read_active_ranks_from_file()
            else:
                new_ranks = None
            new_ranks = hvd.broadcast_object(new_ranks, root_rank=0, name="ranks_check_bcast")
            print(f'updated active rank on {hvd.rank()}')

            if new_ranks != current_active_ranks:
                config_changed = True
                break

            if hvd.rank() in current_active_ranks:
                try:
                    print('Training:')
                    ST_batch = time.time()
                    images, target = next(data_iterator)
                    images, target = images.cuda(), target.cuda()

                    base_optimizer.zero_grad()
                    output = model(images)
                    loss = F.cross_entropy(output, target)
                    ST_back = time.time()
                    loss.backward()
                    # --- FIX: Manual Gradient Allreduce ---
                    # Loop over all model parameters and average their gradients.
                    ST_grad = time.time()
                    # print('allreduce:')
                    # print(active_set)
                    # if active_set is None:
                    #     allreduce_name = "grads_full_world"
                    # else:
                    #     # Create a name unique to this specific subset of ranks
                    #     ranks_str = "_".join(map(str, sorted(current_active_ranks)))
                    #     allreduce_name = f"grads_set_{ranks_str}"

                    # # Pass the unique name to our helper function
                    # allreduce_gradients_manual(model, active_set, name=allreduce_name)
                    for i, param in enumerate(model.parameters()):
                        if param.grad is not None:
                            if active_set is None:
                                hvd.allreduce_(param.grad, average=True, name=f"grad_{i}")
                            else:
                                hvd.allreduce_(param.grad, average=True, process_set=active_set, name=f"grad_{i}")
                    print(f'grad allreduce Cost: {time.time() - ST_grad}s')
                    # Step the optimizer with the averaged gradient
                    ST_step = time.time()
                    base_optimizer.step()
                    print(f'op.step() Cost: {time.time() - ST_step}s')
                    # --- END FIX ---
                    print(f'One Batch Cost: {time.time() - ST_batch}s')
                    
                    state.batch_idx += 1
                    if hvd.rank() == current_active_ranks[0]:
                        print(f"Epoch: {state.epoch} | Batch: {state.batch_idx-1} | Loss: {loss.item():.4f}")
                except StopIteration:
                    break
            else:
                base_optimizer.zero_grad()
                model.cpu()
                torch.cuda.empty_cache()
                time.sleep(1)

        if not config_changed:
            state.epoch += 1
            state.batch_idx = 0

def read_active_ranks_from_file(filepath='/mydata/Data/DynGPUs/custom_hvd/active_workers.txt'):
    # ... (same as before) ...
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
    """
    Manually performs a single, fused allreduce on all of a model's gradients
    using a provided unique name for the operation.
    """
    params_with_grad = [p for p in model.parameters() if p.grad is not None]
    if not params_with_grad:
        return

    flat_grads = torch.cat([p.grad.view(-1) for p in params_with_grad])

    # Pass the unique name to the allreduce call
    if process_set is None:
        hvd.allreduce_(flat_grads, average=True, name=name)
    else:
        hvd.allreduce_(flat_grads, average=True, process_set=process_set, name=name)

    offset = 0
    for p in params_with_grad:
        numel = p.grad.numel()
        p.grad.copy_(flat_grads[offset:offset + numel].view_as(p.grad))
        offset += numel

if __name__ == "__main__":
    main()