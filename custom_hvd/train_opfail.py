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

class TrainingState:
    def __init__(self):
        self.epoch = 0
        self.batch_idx = 0

def main():
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    model = models.resnet50().cuda()
    # Optimizer is created later, only by active ranks
    base_optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss().cuda()
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
                # All workers participate in deciding the new active set
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

                # --- FINAL FIX: Only active ranks participate in state sync and setup ---
                if hvd.rank() in current_active_ranks:
                    data_iterator = None
                    local_rank = current_active_ranks.index(hvd.rank())
                    num_replicas = len(current_active_ranks)
                    
                    sampler = DistributedSampler(train_dataset, num_replicas=num_replicas, rank=local_rank)
                    sampler.set_epoch(state.epoch)
                    loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=4, sampler=sampler)
                    data_iterator = iter(loader)
                    
                    if active_set is None:
                        hvd_optimizer = hvd.DistributedOptimizer(base_optimizer, named_parameters=model.named_parameters())
                    else:
                        hvd_optimizer = hvd.DistributedOptimizer(base_optimizer, named_parameters=model.named_parameters(), process_set=active_set)
                    
                    root_rank_for_sync = current_active_ranks[0] if not is_full_world else 0
                    
                    if hvd.rank() == root_rank_for_sync:
                        model_state = model.state_dict()
                        opt_state = base_optimizer.state_dict()
                    else:
                        model_state, opt_state = None, None
                    
                    if active_set is None:
                        bcast_model_state = hvd.broadcast_object(model_state, root_rank=root_rank_for_sync)
                        bcast_opt_state = hvd.broadcast_object(opt_state, root_rank=root_rank_for_sync)
                        state = hvd.broadcast_object(state, root_rank=root_rank_for_sync)
                    else:
                        bcast_model_state = hvd.broadcast_object(model_state, root_rank=root_rank_for_sync, process_set=active_set)
                        bcast_opt_state = hvd.broadcast_object(opt_state, root_rank=root_rank_for_sync, process_set=active_set)
                        state = hvd.broadcast_object(state, root_rank=root_rank_for_sync, process_set=active_set)

                    if hvd.rank() != root_rank_for_sync:
                        model.load_state_dict(bcast_model_state)
                        base_optimizer.load_state_dict(bcast_opt_state)

                    # Fast-forward the new iterator to the correct batch
                    for _ in range(state.batch_idx):
                        next(data_iterator)
                # --- END FIX ---
                config_changed = False

            # All workers check for config changes to allow paused workers to rejoin
            if hvd.rank() == 0:
                new_ranks = read_active_ranks_from_file()
            else:
                new_ranks = None
            new_ranks = hvd.broadcast_object(new_ranks, root_rank=0, name="ranks_check_bcast")

            if new_ranks != current_active_ranks:
                config_changed = True
                break

            # Only active workers proceed to train
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
                except StopIteration:
                    break
            else:
                # Paused workers do nothing but sleep
                model.cpu()
                time.sleep(1)

        if not config_changed:
            state.epoch += 1
            state.batch_idx = 0

def read_active_ranks_from_file(filepath='/tmp/active_workers.txt'):
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