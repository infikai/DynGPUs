import time
import os

# Attempt to import PyTorch, provide guidance if not found
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.models as models
    import torchvision.transforms as transforms
    from torchvision.datasets import ImageFolder # For loading ImageNet
    from torch.utils.data import DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("WARNING: PyTorch is not installed. This script cannot run.")
    print("Please install PyTorch and torchvision with: pip install torch torchvision")
    # Define dummy classes if PyTorch is not available to prevent NameErrors later
    class Dataset: pass
    class ImageFolder(Dataset): pass


# --- Configuration ---
CHECKPOINT_DIR = "./checkpoints"
IMAGENET_DATA_PATH = "/mydata/Data/imagenet" # User-specified path for ImageNet data

def save_checkpoint(epoch, model, optimizer, loss, filename_prefix="checkpoint"):
    """Saves model checkpoint and measures time taken."""
    if not PYTORCH_AVAILABLE:
        return

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        print(f"[TRAIN] Created checkpoint directory: {CHECKPOINT_DIR}")

    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    # Save a checkpoint for the specific epoch
    epoch_filename = os.path.join(CHECKPOINT_DIR, f"{filename_prefix}_epoch_{epoch}.pth")
    start_time_epoch_save = time.time()
    torch.save(state, epoch_filename)
    end_time_epoch_save = time.time()
    print(f"[TRAIN] Saved epoch checkpoint to {epoch_filename} (took {end_time_epoch_save - start_time_epoch_save:.2f}s)")

    # Overwrite a 'latest' checkpoint for easy resuming
    latest_filename = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
    start_time_latest_save = time.time()
    torch.save(state, latest_filename)
    end_time_latest_save = time.time()
    print(f"[TRAIN] Updated latest checkpoint to {latest_filename} (took {end_time_latest_save - start_time_latest_save:.2f}s)")


def load_checkpoint(model, optimizer, filename="latest_checkpoint.pth"):
    """Loads model checkpoint from the checkpoint directory."""
    if not PYTORCH_AVAILABLE:
        return 0, None

    filepath = os.path.join(CHECKPOINT_DIR, filename)
    if os.path.isfile(filepath):
        print(f"[TRAIN] Loading checkpoint '{filepath}'")
        try:
            # Load checkpoint onto the same device it was saved from
            checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            last_loss = checkpoint.get('loss', None)
            print(f"[TRAIN] Loaded checkpoint '{filepath}' (epoch {checkpoint['epoch']}, loss: {last_loss})")
            print(f"[TRAIN] Resuming training from epoch {start_epoch}")
            return start_epoch, last_loss
        except Exception as e:
            print(f"[TRAIN] Error loading checkpoint {filepath}: {e}. Starting from scratch.")
            return 0, None
    else:
        print(f"[TRAIN] No checkpoint found at '{filepath}'. Starting from scratch.")
        return 0, None


def train():
    """
    Main PyTorch training function for ResNet50 with ImageNet data.
    Handles data loading, training loop, and checkpointing.
    """
    if not PYTORCH_AVAILABLE:
        print("[TRAIN] PyTorch not available. Exiting training.")
        return

    training_start_time = time.time()
    print("[TRAIN] PyTorch Training started with ResNet50 on ImageNet data.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Using device: {device}")

    # --- Model, Optimizer, and Loss Function ---
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # --- Load Checkpoint ---
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    start_epoch, _ = load_checkpoint(model, optimizer)

    # --- Data Loading ---
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_dir = os.path.join(IMAGENET_DATA_PATH, 'train')
    print(f"[TRAIN] Attempting to load ImageNet training data from: {train_dir}")

    try:
        train_dataset = ImageFolder(train_dir, train_transform)
    except FileNotFoundError:
        print(f"[TRAIN] ERROR: ImageNet training data not found at {train_dir}. Please check the path in the script.")
        print("[TRAIN] Exiting training function due to missing data.")
        return
    except Exception as e:
        print(f"[TRAIN] ERROR: Could not load ImageNet dataset: {e}")
        print("[TRAIN] Exiting training function.")
        return

    if len(train_dataset) == 0:
        print(f"[TRAIN] ERROR: ImageNet training dataset at {train_dir} is empty. Ensure the directory structure is correct (e.g., {train_dir}/class_name/image.JPEG).")
        print("[TRAIN] Exiting training function.")
        return

    print(f"[TRAIN] Successfully loaded {len(train_dataset)} images from ImageNet training set.")

    # Adjust workers and batch size based on whether a GPU is used
    num_dataloader_workers = 1 if device.type == 'cuda' else 0
    batch_size = 96 if device.type == 'cuda' else 32
    print(f"[TRAIN] DataLoader using num_workers={num_dataloader_workers}, batch_size={batch_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_dataloader_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # --- Training Loop ---
    max_epochs = 100
    epoch_for_interrupt_save = start_epoch
    
    print(f"--- Starting training loop from epoch {start_epoch} ---")

    try:
        model.train() # Set model to training mode

        for epoch in range(start_epoch, max_epochs):
            epoch_for_interrupt_save = epoch
            print(f"[TRAIN] Epoch {epoch+1}/{max_epochs}")
            epoch_loss_aggregator = 0.0
            num_batches_in_epoch = 0
            epoch_start_time = time.time()

            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                # Standard training steps
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss_aggregator += loss.item()
                num_batches_in_epoch += 1

                # Log progress periodically
                if (i + 1) % 100 == 0:
                    print(f"[TRAIN] Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Current Batch Loss: {loss.item():.4f}")

            # Calculate and log average loss for the epoch
            avg_epoch_loss = epoch_loss_aggregator / num_batches_in_epoch if num_batches_in_epoch > 0 else 0
            epoch_end_time = time.time()
            print(f"[TRAIN] Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f} (took {epoch_end_time - epoch_start_time:.2f}s)")

            # Save a checkpoint at the end of each epoch
            save_checkpoint(epoch, model, optimizer, avg_epoch_loss)

        print(f"[TRAIN] Training completed after reaching max_epochs ({max_epochs}).")

    except KeyboardInterrupt:
        print("\n[TRAIN] KeyboardInterrupt received. Saving final checkpoint...")
        # Loss value is not calculated for the interrupted epoch, so we save with None
        print(f"[TRAIN] Attempting to save interrupt checkpoint for epoch {epoch_for_interrupt_save}...")
        save_checkpoint(epoch_for_interrupt_save, model, optimizer, None, filename_prefix="interrupt_checkpoint")
        print("[TRAIN] Training function cleanup complete after interrupt. Exiting.")
    except Exception as e:
        print(f"[TRAIN] An unexpected error occurred during training: {e}")
        try:
            print(f"[TRAIN] Attempting to save a crash checkpoint for epoch {epoch_for_interrupt_save}...")
            save_checkpoint(epoch_for_interrupt_save, model, optimizer, None, filename_prefix="crash_checkpoint")
        except Exception as ce:
            print(f"[TRAIN] Could not save crash checkpoint: {ce}")
    finally:
        total_training_time = time.time() - training_start_time
        print(f"[TRAIN] Training function finished. Total run time: {total_training_time:.2f}s")


if __name__ == "__main__":
    # Main execution block
    if not PYTORCH_AVAILABLE:
        print("MAIN: PyTorch is not installed. Please install it to run this script.")
    else:
        print("MAIN: Starting training script.")
        print(f"MAIN: Attempting to use ImageNet data from: {IMAGENET_DATA_PATH}")
        print(f"MAIN: Please ensure '{os.path.join(IMAGENET_DATA_PATH, 'train')}' exists and is structured for PyTorch's ImageFolder.")
        print(f"MAIN: Checkpoints will be saved in: {os.path.abspath(CHECKPOINT_DIR)}")
        print("MAIN: Press Ctrl+C to interrupt training and save a checkpoint.")
        
        # Directly call the training function
        train()

        print("MAIN: Program terminated.")