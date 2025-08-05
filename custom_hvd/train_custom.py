import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import horovod.torch as hvd
import time
import os
from torch.utils.data.distributed import DistributedSampler
import socket

# A simple class to hold our synchronized state
class TrainingState:
    def __init__(self):
        self.epoch = 0
        self.batch_idx = 0

def main():
    # --- 1. Boilerplate Setup ---
    hvd.init(process_sets="dynamic")
    hostname = socket.gethostname()
    print(f'Node: {hostname} binded rank is {hvd.rank()}')
    torch.cuda.set_device(hvd.local_rank())

    torch.manual_seed(11)
    torch.cuda.manual_seed(11)

    train_dataset = datasets.ImageFolder(
        os.path.join('/mydata/Data/imagenet', 'train'),
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))

    val_dataset = datasets.ImageFolder(
        os.path.join('/mydata/Data/imagenet', 'val'),
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ]))

    state = TrainingState()
    current_active_ranks = []

    model = models.resnet50().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss().cuda()

    process_set_cache = {}
    
    # --- 2. Main Training Controller ---
    while state.epoch < 100:
        config_changed = True
        
        # This inner loop processes batches and can be broken and restarted
        while True:
            # --- Rebuild communication and data pipeline if config changed ---
            if config_changed:
                if hvd.rank() == 0:
                    active_ranks = read_active_ranks_from_file()
                    print('Active ranks:', active_ranks)
                else:
                    active_ranks = None
                active_ranks = hvd.broadcast_object(active_ranks, root_rank=0, name="ranks_bcast")

                current_active_ranks = active_ranks
                # Use a tuple of ranks as a key for our cache.
                is_full_world = (len(current_active_ranks) == hvd.size())

                if is_full_world:
                    # Case 1: All workers are active. Use the default communicator.
                    print(f"Rank {hvd.rank()}: Configuring for FULL WORLD.")
                    active_set = None # None signifies the default "world" process set
                else:
                    # Case 2: A subset of workers is active. Safely create/get the subset.
                    ranks_tuple = tuple(sorted(current_active_ranks))
                    needs_creation_local = torch.tensor(1 if ranks_tuple not in process_set_cache else 0)
                    needs_creation_sum = hvd.allreduce(needs_creation_local, name=f"creation_lock_{ranks_tuple}")

                    if needs_creation_sum.item() > 0:
                        print(f"Rank {hvd.rank()}: Synchronized decision to create NEW process set for {ranks_tuple}")
                        process_set_cache[ranks_tuple] = hvd.add_process_set(current_active_ranks)
                    
                    active_set = process_set_cache[ranks_tuple]

                data_iterator = None
                if hvd.rank() in current_active_ranks:
                    # Assign rank number by the index in active rank list
                    local_rank = current_active_ranks.index(hvd.rank())
                    sampler = DistributedSampler(train_dataset, num_replicas=len(current_active_ranks), rank=local_rank)
                    sampler.set_epoch(state.epoch)
                    loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, sampler=sampler)
                    data_iterator = iter(loader)
                    # Fast-forward the new iterator to the correct batch
                    # To do: implement something like horovod distributed sampler which can record the processed indices.
                    for _ in range(state.batch_idx):
                        next(data_iterator)

                hvd_optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), process_set=active_set)
                hvd.broadcast_parameters(model.state_dict(), root_rank=current_active_ranks[0], process_set=active_set)
                hvd.broadcast_optimizer_state(optimizer, root_rank=current_active_ranks[0], process_set=active_set)
                state = hvd.broadcast_object(state, root_rank=current_active_ranks[0], name="state_bcast", process_set=active_set)
                config_changed = False

            # --- Check for new config changes before every batch ---
            if hvd.rank() == 0:
                new_ranks = read_active_ranks_from_file()
            else:
                new_ranks = None
            new_ranks = hvd.broadcast_object(new_ranks, root_rank=0, name="ranks_check_bcast")

            if new_ranks != current_active_ranks:
                config_changed = True
                break # Break batch loop to force reconfiguration

            # --- Process one batch ---
            if hvd.rank() in current_active_ranks:
                try:
                    images, target = next(data_iterator)
                    model.cuda()
                    images, target = images.cuda(), target.cuda()
                    
                    hvd_optimizer.zero_grad()
                    output = model(images)
                    loss = criterion(output, target)
                    loss.backward()
                    hvd_optimizer.step()
                    
                    state.batch_idx += 1
                    if hvd.rank() == current_active_ranks[0]:
                        print(f"Epoch: {state.epoch} | Batch: {state.batch_idx-1} | Loss: {loss.item():.4f}")

                except StopIteration:
                    # Reached the end of the epoch successfully
                    break
            else:
                # Paused worker waits
                model.cpu()
                torch.cuda.empty_cache()
                time.sleep(1)

        # --- End of Epoch Logic ---
        if not config_changed:
            # If we finished the epoch without interruption, advance to the next one
            state.epoch += 1
            state.batch_idx = 0

def read_active_ranks_from_file(filepath='/mydata/Data/DynGPUS/custom_hvd/active_workers.txt'):
    """Reads a comma-separated list of active ranks from the control file."""
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

if __name__ == "__main__":
    main()