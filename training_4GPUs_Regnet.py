import os
import time
import argparse
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, models, transforms

def setup(rank, world_size):
    """Initializes the distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
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
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, sampler=val_sampler)

    return train_loader, val_loader

def main(rank, world_size, args):
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)

    # Model Initialization
    model = models.regnet_y_128gf(weights=None)
    model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.SGD(ddp_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Data Loading
    train_loader, val_loader = get_imagenet_dataloaders(args.data_dir, args.batch_size, args.workers, world_size, rank)

    # Training Loop
    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        ddp_model.train()
        epoch_start_time = time.time()

        for i, (images, target) in enumerate(train_loader):
            batch_load_start = time.time()
            images = images.to(rank, non_blocking=True)
            target = target.to(rank, non_blocking=True)
            batch_load_end = time.time()

            # Forward and backward pass
            train_step_start = torch.cuda.Event(enable_timing=True)
            train_step_end = torch.cuda.Event(enable_timing=True)

            train_step_start.record()
            output = ddp_model(images)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_step_end.record()
            torch.cuda.synchronize()

            if rank == 0 and i % args.print_freq == 0:
                batch_load_time = (batch_load_end - batch_load_start) * 1000
                train_time = train_step_start.elapsed_time(train_step_end)
                print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                      f'Loss {loss.item():.4f}\t'
                      f'Batch Loading Time: {batch_load_time:.2f}ms\t'
                      f'Batch Training Time: {train_time:.2f}ms')

        scheduler.step()

        # Checkpointing
        if rank == 0 and (epoch + 1) % args.checkpoint_freq == 0:
            checkpoint_start = time.time()
            torch.save({
                'epoch': epoch + 1,
                'state_dict': ddp_model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, f'checkpoint_epoch_{epoch+1}.pth.tar')
            checkpoint_end = time.time()
            print(f"Epoch {epoch+1} checkpoint saved. Time taken: {(checkpoint_end - checkpoint_start) * 1000:.2f}ms")

        # Validation (optional, can be added here)

    if rank == 0:
        print(f"Total training time: {datetime.timedelta(seconds=time.time() - epoch_start_time)}")

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data-dir', default='/mydata/Data/imagenet', type=str, help='path to dataset')
    parser.add_argument('--epochs', default=90, type=int, help='number of total epochs to run')
    parser.add_argument('--batch-size', default=8, type=int, help='mini-batch size per GPU')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--workers', default=16, type=int, help='number of data loading workers')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--checkpoint-freq', default=1, type=int, help='checkpoint frequency in epochs')
    parser.add_argument('--world-size', default=4, type=int, help='number of GPUs')
    args = parser.parse_args()

    # Use torch.multiprocessing.spawn to launch distributed processes
    mp.spawn(main, args=(args.world_size, args), nprocs=args.world_size, join=True)