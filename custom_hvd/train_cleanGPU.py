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

# Hyperparameters
EPOCHS = 100
BATCH_SIZE = 64

class TrainingState:
    def __init__(self):
        self.epoch = 0
        self.batch_idx = 0
        self.processed_num = 0
        self.seed = 0

def main():
    hvd.init(process_sets="dynamic")
    hostname = socket.gethostname()
    parts = hostname.split('.')
    nodename = parts[0]
    print(f'{hostname} binded to horovod rank{hvd.rank()}.')

    torch.cuda.set_device(hvd.local_rank())

    # Two experiment model to test
    ST_model = time.time()
    # model = models.resnet50().cuda()
    model = models.vit_l_32(weights=None).cuda()

    base_optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
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

    # Train
    while state.epoch < EPOCHS:
        config_changed = True

        while True:
            if config_changed:
                # Time how long config took
                ST_config = time.time()

                # This logic for determining the active set remains the same
                if hvd.rank() == 0:
                    active_ranks = read_active_ranks_from_file()
                else:
                    active_ranks = None
                active_ranks = hvd.broadcast_object(active_ranks, root_rank=0, name="ranks_bcast")

                old_active_ranks = current_active_ranks
                print(f'Old ranks: {old_active_ranks}')
                current_active_ranks = active_ranks
                print(f'Old ranks: {current_active_ranks}')
                is_full_world = (len(current_active_ranks) == hvd.size())

                if hvd.rank() not in old_active_ranks:
                    print(hvd.rank())
                    move_optimizer_state(base_optimizer, 'gpu')

                # Two case to determining
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
                    # Move model to GPU
                    model.cuda()
                    root_rank_for_sync = 0
                    # Sync Model and Optimizer
                    ST_bcast = time.time()
                    if is_full_world:
                        print('=== Full world case ===')
                        # Case 1: All workers active. Use the built-in functions.
                        hvd.broadcast_parameters(model.state_dict(), root_rank=root_rank_for_sync)
                        print(f'Model Bcast Cost: {time.time() - ST_bcast}s')

                        ST_OP = time.time()
                        hvd.broadcast_optimizer_state(base_optimizer, root_rank=root_rank_for_sync)
                        print(f'OP Bcast Cost: {time.time() - ST_OP}s')

                        ST_state = time.time()
                        state = hvd.broadcast_object(state, root_rank=root_rank_for_sync, name="BcastState")
                        print(f'State Bcast Cost: {time.time() - ST_state}s')

                        print(f'Whole BCAST cost: {time.time() - ST_bcast}s')
                    else:
                        print('=== Partial world case ===')
                        # Case 2: A subset is active. Use the altered broadcast function.
                        hvd.broadcast_parameters(model.state_dict(), root_rank=root_rank_for_sync, process_set=active_set)
                        print(f'Model Bcast Cost: {time.time() - ST_bcast}s')

                        ST_OP = time.time()
                        hvd.broadcast_optimizer_state(base_optimizer, root_rank=root_rank_for_sync, process_set=active_set)
                        print(f'OP Bcast Cost: {time.time() - ST_OP}s')

                        ST_state = time.time()
                        state = hvd.broadcast_object(state, root_rank=root_rank_for_sync, process_set=active_set, name="BcastState")
                        print(f'State Bcast Cost: {time.time() - ST_state}s')
                        
                        print(f'Whole BCAST cost: {time.time() - ST_bcast}s')
                    print('==='*5)

                    local_rank = current_active_ranks.index(hvd.rank())
                    ST_sampler = time.time()
                    sampler = MyElasticSampler(train_dataset)
                    sampler.set_epoch(state.epoch, state.processed_num, num_replicas=len(current_active_ranks), rank=local_rank)
                    print(f'Sampler Cost: {time.time() - ST_sampler}s')

                    ST_loader = time.time()
                    loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, sampler=sampler)
                    print(f'Loader Cost: {time.time() - ST_loader}s')
                    data_iterator = iter(loader)

                print(f'Config Change Cost: {time.time() - ST_config}s')
                config_changed = False

            if hvd.rank() == 0:
                new_ranks = read_active_ranks_from_file()
            else:
                new_ranks = None
            new_ranks = hvd.broadcast_object(new_ranks, root_rank=0, name="ranks_check_bcast")
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

                    ST_backward = time.time()
                    loss.backward()
                    print(f'Backward pass Cost: {time.time() - ST_backward}s')

                    ST_grad = time.time()
                    if active_set is None:
                        allreduce_name = "grads_full_world"
                    else:
                        # Create a name unique to this specific subset of ranks
                        ranks_str = "_".join(map(str, sorted(current_active_ranks)))
                        allreduce_name = f"grads_set_{ranks_str}"

                    # Pass the unique name to our helper function
                    allreduce_gradients_manual(model, active_set, name=allreduce_name)
                    print(f'gradients allreduce Cost: {time.time() - ST_grad}s')

                    # Step the optimizer with the averaged gradient
                    ST_step = time.time()
                    base_optimizer.step()
                    print(f'op.step() Cost: {time.time() - ST_step}s')

                    print(f'One Batch Cost: {time.time() - ST_batch}s')
                    sampler.record_batch(state.batch_idx, BATCH_SIZE)
                    state.batch_idx += 1
                    state.processed_num = sampler.get_processed_num()
                    if hvd.rank() == current_active_ranks[0]:
                        print(f"Epoch: {state.epoch} | Batch: {state.batch_idx-1} | Loss: {loss.item():.4f}")
                except StopIteration:
                    break
            else:
                base_optimizer.zero_grad()
                move_optimizer_state(base_optimizer, 'cpu')
                model.cpu()
                torch.cuda.empty_cache()
                time.sleep(1)

        # epoch end
        if not config_changed:
            state.epoch += 1
            state.batch_idx = 0
            state.processed_num = 0
            sampler.set_epoch(state.epoch, state.processed_num)

def read_active_ranks_from_file(filepath='/mydata/Data/DynGPUs/custom_hvd/active_workers.txt'):
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

def move_optimizer_state(optimizer, device):
    """Moves the state of the optimizer to the specified device."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

if __name__ == "__main__":
    main()