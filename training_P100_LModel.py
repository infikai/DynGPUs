import os
import time
import argparse
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
# Import Automatic Mixed Precision utilities
from torch.cuda.amp import GradScaler, autocast

def get_imagenet_dataloaders(data_dir, batch_size, num_workers):
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

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,  # Use shuffle for single GPU
        num_workers=num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader

def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training on a Single P100')
    parser.add_argument('--data-dir', default='/mydata/Data/imagenet', type=str, help='path to dataset')
    parser.add_argument('--epochs', default=90, type=int, help='number of total epochs to run')
    # CRITICAL: Batch size must be very small for a 12GB GPU. May need to be reduced to 1 or 2.
    parser.add_argument('--batch-size', default=4, type=int, help='mini-batch size per GPU')
    # CRITICAL: Accumulate gradients to simulate a larger batch size.
    parser.add_argument('--accumulation-steps', default=16, type=int, help='number of steps to accumulate gradients over')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--checkpoint-freq', default=1, type=int, help='checkpoint frequency in epochs')
    args = parser.parse_args()

    # --- Device Setup ---
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA-capable GPU.")
    device = torch.device("cuda:0")
    print(f"Using device: {torch.cuda.get_device_name(0)}")

    # --- Model Initialization ---
    model = models.regnet_y_128gf(weights=None)
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()

    # --- Data Loading ---
    train_loader, _ = get_imagenet_dataloaders(args.data_dir, args.batch_size, args.workers)
    
    # Calculate the effective batch size
    effective_batch_size = args.batch_size * args.accumulation_steps
    print(f"Physical batch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")


    # --- Training Loop ---
    start_train_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad() # Zero gradients once at the start of the accumulation phase

        enumerate_init_start = time.time()
        for i, (images, target) in enumerate(train_loader):
            batch_load_start = time.time()
            if i == 0:
                print(f'Enumerate init Time: {batch_load_start - enumerate_init_start:.2f}s')
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            batch_load_time = (time.time() - batch_load_start) * 1000

            train_step_start = torch.cuda.Event(enable_timing=True)
            train_step_end = torch.cuda.Event(enable_timing=True)

            train_step_start.record()

            # Use autocast for the forward pass
            with autocast():
                output = model(images)
                # Normalize loss for accumulation
                loss = criterion(output, target) / args.accumulation_steps

            # Scale the loss and call backward() to create scaled gradients
            scaler.scale(loss).backward()
            
            # --- Gradient Accumulation Step ---
            # Update weights only after accumulating gradients for accumulation_steps
            if (i + 1) % args.accumulation_steps == 0:
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer)
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called.
                scaler.step(optimizer)
                # Updates the scale for next iteration.
                scaler.update()
                # Zero out the gradients for the next accumulation phase
                optimizer.zero_grad()


            train_step_end.record()
            torch.cuda.synchronize()

            if i % args.print_freq == 0:
                train_time = train_step_start.elapsed_time(train_step_end)
                # Note: The printed loss is the scaled-down version
                print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                      f'Loss {loss.item() * args.accumulation_steps:.4f}\t' # Scale up for logging
                      f'Batch Loading Time: {batch_load_time:.2f}ms\t'
                      f'Batch Training Time: {train_time:.2f}ms')

        scheduler.step()

        # --- Checkpointing ---
        if (epoch + 1) % args.checkpoint_freq == 0:
            checkpoint_start = time.time()
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(), # No .module here
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(), # Save the scaler state
            }, f'checkpoint_epoch_{epoch+1}.pth.tar')
            checkpoint_time = (time.time() - checkpoint_start) * 1000
            print(f"Epoch {epoch+1} checkpoint saved. Time taken: {checkpoint_time:.2f}ms")

    total_training_time = datetime.timedelta(seconds=int(time.time() - start_train_time))
    print(f"Total training time: {total_training_time}")


if __name__ == '__main__':
    main()