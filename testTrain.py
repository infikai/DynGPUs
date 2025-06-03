import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import os
import time # For timing epochs

# --- Configuration ---
DATA_DIR = '/mydata/Data/imagenet' # IMPORTANT: Set this to your ImageNet directory
MODEL_NAME = 'resnet50'
NUM_CLASSES = 1000 # ImageNet has 1000 classes
BATCH_SIZE = 64  # Adjust based on your GPU memory
NUM_WORKERS = 10  # Adjust based on your CPU cores
LEARNING_RATE = 0.01 # Initial learning rate
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
EPOCHS = 90 # Standard for ImageNet from scratch, adjust as needed
CHECKPOINT_PATH = f"{MODEL_NAME}_imagenet_checkpoint.pth"
CHECKPOINT_INTERVAL = 5 # Save checkpoint every epoch (good for long runs)
PRINT_FREQ = 100 # Print training stats every 100 batches

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data Loading and Preprocessing ---
# Normalization parameters for ImageNet
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Training data transformations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

# Validation data transformations
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

# Create DataLoaders
# Make sure your ImageNet data is structured correctly:
# DATA_DIR/train/n01440764/ILSVRC2012_val_00000293.JPEG
# DATA_DIR/val/n01440764/ILSVRC2012_val_00000293.JPEG
try:
    train_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, 'train'),
        train_transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, 'val'),
        val_transform
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    print(f"Loaded training data: {len(train_dataset)} images")
    print(f"Loaded validation data: {len(val_dataset)} images")

except FileNotFoundError:
    print(f"ERROR: ImageNet data not found at {DATA_DIR}.")
    print("Please download ImageNet manually and set the DATA_DIR variable.")
    exit()
except Exception as e:
    print(f"Error loading ImageNet data: {e}")
    exit()


# --- Model Definition ---
print(f"Loading model: {MODEL_NAME}")
# Load ResNet50. You can choose pretrained=True if you want to fine-tune
# or pretrained=False to train from scratch.
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if False else None, progress=True) # Set to True to use pretrained weights

# If training from scratch or adapting to a different number of classes (not needed for standard ImageNet)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

model = model.to(device)

# If using multiple GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)


# --- Optimizer and Loss Function ---
optimizer = optim.SGD(model.parameters(), LEARNING_RATE,
                      momentum=MOMENTUM,
                      weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss().to(device)

# Learning rate scheduler (optional, but common for ImageNet)
# Example: StepLR, reduce learning rate by a factor of 10 every 30 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


# --- Load Checkpoint (if exists) ---
start_epoch = 0
best_val_accuracy = 0.0 # Track best validation accuracy

if os.path.exists(CHECKPOINT_PATH):
    print(f"Loading checkpoint from '{CHECKPOINT_PATH}'")
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True) # Load to current device
        
        # Adjust for DataParallel: remove 'module.' prefix if saved with it and loading without it, or vice-versa
        model_state_dict = checkpoint['model_state_dict']
        if isinstance(model, nn.DataParallel) and not list(model_state_dict.keys())[0].startswith('module.'):
            model_state_dict = {'module.' + k: v for k, v in model_state_dict.items()}
        elif not isinstance(model, nn.DataParallel) and list(model_state_dict.keys())[0].startswith('module.'):
            model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
        
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Resuming training from epoch {start_epoch}. Best validation accuracy: {best_val_accuracy:.4f}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}. Starting from scratch.")
        start_epoch = 0
        best_val_accuracy = 0.0
else:
    print("No checkpoint found, starting training from scratch.")


# --- Training and Validation Functions ---
def train_one_epoch(epoch):
    model.train()
    epoch_loss = 0
    epoch_corrects = 0
    total_samples = 0
    start_time = time.time()

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        epoch_loss += loss.item() * inputs.size(0)
        epoch_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)

        if batch_idx % PRINT_FREQ == 0 or batch_idx == len(train_loader) -1 :
            current_loss = loss.item()
            current_acc = torch.sum(preds == labels.data).double() / inputs.size(0)
            print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {current_loss:.4f} Acc: {current_acc:.4f}")

    epoch_loss /= total_samples
    epoch_acc = epoch_corrects.double() / total_samples
    epoch_time = time.time() - start_time
    print(f"--- Training Epoch {epoch+1} ---")
    print(f"Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Time: {epoch_time:.2f}s")
    return epoch_loss, epoch_acc

def validate_one_epoch():
    model.eval()
    val_loss = 0
    val_corrects = 0
    total_samples = 0
    start_time = time.time()

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

    val_loss /= total_samples
    val_acc = val_corrects.double() / total_samples
    epoch_time = time.time() - start_time
    print(f"--- Validation ---")
    print(f"Loss: {val_loss:.4f} Acc: {val_acc:.4f} Time: {epoch_time:.2f}s")
    return val_loss, val_acc

# --- Training Loop ---
print("Starting training...")
try:
    for epoch in range(start_epoch, EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        train_loss, train_acc = train_one_epoch(epoch)
        val_loss, val_acc = validate_one_epoch()

        if scheduler is not None:
            scheduler.step() # Step the learning rate scheduler

        # --- Save Checkpoint ---
        is_best = val_acc > best_val_accuracy
        if is_best:
            best_val_accuracy = val_acc
            print(f"New best validation accuracy: {best_val_accuracy:.4f}. Saving checkpoint...")
        
        # Save checkpoint periodically or if it's the best model
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0 or is_best:
            if not is_best:
                 print(f"Saving checkpoint at epoch {epoch+1}...")
            
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(), # Use model.module.state_dict() if using DataParallel and want to save without the 'module.' prefix
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_accuracy': best_val_accuracy,
                'current_val_accuracy': val_acc,
                'current_val_loss': val_loss,
            }
            if scheduler is not None:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            
            torch.save(checkpoint_data, CHECKPOINT_PATH)
            if is_best:
                torch.save(checkpoint_data, f"{MODEL_NAME}_imagenet_best.pth") # Save best model separately

except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving current progress (if any unsaved)...")
    # The periodic/best model saving should cover most cases.
    # You could add a final save here if needed, but ensure it's the state
    # from the *last completed epoch* or be careful about inconsistent state.
    print("Exiting.")
    # Optionally save one last time
    checkpoint_data = {
        'epoch': epoch, # This might be a partially completed epoch
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_accuracy': best_val_accuracy,
    }
    if scheduler is not None:
        checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(checkpoint_data, CHECKPOINT_PATH) # + "_interrupt")

except Exception as e:
    print(f"An error occurred during training: {e}")
    import traceback
    traceback.print_exc()
    # Optionally save a crash checkpoint
    # print("Attempting to save a crash checkpoint...")
    # crash_checkpoint_path = CHECKPOINT_PATH + "_crash"
    # checkpoint_data = {
    #     'epoch': epoch if 'epoch' in locals() else -1,
    #     'model_state_dict': model.state_dict() if 'model' in locals() else None,
    #     'optimizer_state_dict': optimizer.state_dict() if 'optimizer' in locals() else None,
    #     'best_val_accuracy': best_val_accuracy if 'best_val_accuracy' in locals() else 0.0,
    #     'error_message': str(e)
    # }
    # if scheduler is not None and 'scheduler' in locals():
    #     checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
    # torch.save(checkpoint_data, crash_checkpoint_path)
    # print(f"Crash checkpoint saved to {crash_checkpoint_path}")

finally:
    print("Finished Training (or Canceled).")
    print(f"Best validation accuracy achieved: {best_val_accuracy:.4f}")
