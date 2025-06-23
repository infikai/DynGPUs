import os
import time
import argparse
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, models, transforms

def setup(rank, world_size):
    """Initializes the distributed process group."""
    # These variables are set by torchrun
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12355')
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

def get_imagenet_dataloaders(data_dir, batch_size, num_workers, world_size, rank):
    """Creates ImageNet dataloaders for training and validation."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'train'),
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'val'),
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, # shuffle is mutually exclusive with sampler
        num_workers=num_workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, sampler=val_sampler)

    return train_loader, val_loader

def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data-dir', default='/mydata/Data/imagenet', type=str, help='path to dataset')
    parser.add_argument('--epochs', default=90, type=int, help='number of total epochs to run')
    parser.add_argument('--batch-size', default=256, type=int, help='mini-batch size per GPU')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--workers', default=16, type=int, help='number of data loading workers')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--checkpoint-freq', default=1, type=int, help='checkpoint frequency in epochs')
    args = parser.parse_args()

    # --- Distributed Training Setup ---
    # These environment variables are set by torchrun
    try:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
    except KeyError:
        raise RuntimeError("This script must be launched with torchrun.")

    # Set the device for this process. This is the GPU that this process will use.
    torch.cuda.set_device(local_rank)
    
    # Initialize the distributed environment.
    setup(rank, world_size)
    print(f"Starting DDP on rank {rank} (local rank {local_rank}) of {world_size} processes.")

    # --- Model Initialization ---
    device = torch.device(f"cuda:{local_rank}")
    model = models.vit_h_14(weights=None)
    model.to(device)
    
    # Wrap the model with DDP.
    # find_unused_parameters can be helpful for debugging but incurs a small overhead.
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(ddp_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # --- Data Loading ---
    train_loader, _ = get_imagenet_dataloaders(args.data_dir, args.batch_size, args.workers, world_size, rank)

    # --- Training Loop ---
    start_train_time = time.time()
    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        ddp_model.train()
        
        for i, (images, target) in enumerate(train_loader):
            batch_load_start = time.time()
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            batch_load_time = (time.time() - batch_load_start) * 1000

            # Create CUDA events for accurate timing of GPU operations
            train_step_start = torch.cuda.Event(enable_timing=True)
            train_step_end = torch.cuda.Event(enable_timing=True)

            train_step_start.record()
            
            # Forward pass
            output = ddp_model(images)
            loss = criterion(output, target)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_step_end.record()
            # Waits for everything to finish and syncs
            torch.cuda.synchronize()

            # Only the master process (rank 0) should print logs
            if rank == 0 and i % args.print_freq == 0:
                train_time = train_step_start.elapsed_time(train_step_end)
                print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                      f'Loss {loss.item():.4f}\t'
                      f'Batch Loading Time: {batch_load_time:.2f}ms\t'
                      f'Batch Training Time: {train_time:.2f}ms')

        scheduler.step()

        # --- Checkpointing ---
        # Only rank 0 should save checkpoints to avoid race conditions
        if rank == 0 and (epoch + 1) % args.checkpoint_freq == 0:
            checkpoint_start = time.time()
            # It's recommended to save the model's state_dict from the underlying module
            # not the DDP-wrapped model.
            torch.save({
                'epoch': epoch + 1,
                'state_dict': ddp_model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, f'checkpoint_epoch_{epoch+1}.pth.tar')
            checkpoint_time = (time.time() - checkpoint_start) * 1000
            print(f"Epoch {epoch+1} checkpoint saved. Time taken: {checkpoint_time:.2f}ms")

    if rank == 0:
        total_training_time = datetime.timedelta(seconds=int(time.time() - start_train_time))
        print(f"Total training time: {total_training_time}")

    cleanup()

if __name__ == '__main__':
    main()