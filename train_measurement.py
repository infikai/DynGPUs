import os
import sys
import time
import argparse
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """Saves checkpoint to disk."""
    checkpoint_dir = os.path.dirname(filename)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


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
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    return train_loader

def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training with Resume and Graceful Exit')
    parser.add_argument('--model', type=str, default='regnet_y_128gf',
                        choices=['resnet50', 'regnet_y_128gf'],
                        help='Model architecture to train')
    parser.add_argument('--data-dir', default='/mydata/Data/imagenet', type=str, help='Path to dataset')
    parser.add_argument('--epochs', default=90, type=int, help='Number of total epochs to run')
    parser.add_argument('--batch-size', default=None, type=int, 
                        help='Batch size. Directly impacts memory. If not set, a safe default is chosen based on the model.')
    parser.add_argument('--lr', default=0.1, type=float, help='Initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='Weight decay')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers')
    parser.add_argument('--print-freq', default=10, type=int, help='Print frequency')
    parser.add_argument('--checkpoint-dir', default='checkpoints', type=str, help='Directory to save checkpoints')
    # --- New arguments for resuming ---
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Path to latest checkpoint to resume from (default: none)')
    args = parser.parse_args()

    # Set model-specific defaults if not provided
    if args.batch_size is None:
        args.batch_size = 1 if args.model == 'regnet_y_128gf' else 32
            
    # Device Setup
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA-capable GPU.")
    device = torch.device("cuda:0")
    print(f"Using device: {torch.cuda.get_device_name(0)}")

    # Model Initialization
    print(f"Initializing model: {args.model}")
    model = models.resnet50(weights=None) if args.model == 'resnet50' else models.regnet_y_128gf(weights=None)
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    start_epoch = 0

    # --- Checkpoint Loading Logic ---
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Loading checkpoint '{args.resume}'")
            # Load checkpoint onto the same device it was saved from or the current device
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            
            # Check if the model architecture in the checkpoint matches the current model
            if checkpoint.get('model') and checkpoint['model'] != args.model:
                 print(f"Warning: Checkpoint was saved for model {checkpoint['model']}, but current model is {args.model}. Not loading weights.")
            else:
                model.load_state_dict(checkpoint['state_dict'])

            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"=> Loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            print(f"=> No checkpoint found at '{args.resume}'")

    # Data Loading
    train_loader = get_imagenet_dataloaders(args.data_dir, args.batch_size, args.workers)
    
    print(f"Physical batch size: {args.batch_size}")
    print(f"Training will start from epoch {start_epoch + 1}")

    # --- Training Loop with Interrupt Handler ---
    start_train_time = time.time()
    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            for i, (images, target) in enumerate(train_loader):
                batch_load_start = time.time()
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                batch_load_time = (time.time() - batch_load_start) * 1000

                train_step_start = torch.cuda.Event(enable_timing=True)
                train_step_end = torch.cuda.Event(enable_timing=True)

                train_step_start.record()
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_step_end.record()
                torch.cuda.synchronize()

                if i % args.print_freq == 0:
                    train_time = train_step_start.elapsed_time(train_step_end)
                    print(f'Epoch: [{epoch + 1}][{i}/{len(train_loader)}]\t'
                          f'Loss {loss.item():.4f}\t'
                          f'Batch Loading Time: {batch_load_time:.2f}ms\t'
                          f'Batch Training Time: {train_time:.2f}ms')

            scheduler.step()

            # --- Periodic Checkpointing ---
            checkpoint_filename = os.path.join(args.checkpoint_dir, f'checkpoint_{args.model}_epoch_{epoch+1}.pth.tar')
            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.model,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, filename=checkpoint_filename)

    except KeyboardInterrupt:
        print("\n--- Caught KeyboardInterrupt, saving final checkpoint ---")
        interrupt_filename = os.path.join(args.checkpoint_dir, f'INTERRUPT_{args.model}_epoch_{epoch+1}.pth.tar')
        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.model,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, filename=interrupt_filename)
        print("--- Exiting gracefully ---")
        sys.exit(0)

    total_training_time = datetime.timedelta(seconds=int(time.time() - start_train_time))
    print(f"Total training time: {total_training_time}")

if __name__ == '__main__':
    main()

# python train_measurement.py --model resnet50 --batch-size 64 --resume checkpoints/INTERRUPT_resnet50_epoch_1.pth.tar
# python train_measurement.py --model regnet_y_128gf --batch-size 1 --resume checkpoints/INTERRUPT_regnet_y_128gf_epoch_1.pth.tar