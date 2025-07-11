import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# --- 1. Set up data loading ---
# Define transformations for the training and validation sets
# These are standard transformations for ImageNet
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Set the path to your ImageNet data
data_dir = '/mydata/Data/imagenet'

# Create datasets for training and validation
image_datasets = {x: datasets.ImageFolder(f"{data_dir}/{x}", data_transforms[x])
                  for x in ['train', 'val']}

# Create data loaders to load data in batches
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# --- 2. Define the model ---
# Load a pre-trained ResNet-50 model and reset the final fully connected layer
# for training from scratch.
# If you want to train from scratch, set `pretrained=False`.
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features

# ImageNet has 1000 classes
model.fc = nn.Linear(num_ftrs, 1000)

# --- 3. Set device ---
# Use CPU for training
device = torch.device("cpu")
model = model.to(device)

# --- 4. Define loss function and optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# --- 5. Training loop ---
def train_model(model, criterion, optimizer, num_epochs=1):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if i % 100 == 0:
                    print(f"  Batch {i} of {len(dataloaders[phase])}")


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model

# --- 6. Start training ---
# This will be very slow on a CPU.
# A single epoch can take many hours or days.
model_trained = train_model(model, criterion, optimizer, num_epochs=1)

# --- 7. Save the model ---
torch.save(model_trained.state_dict(), 'resnet50_imagenet_cpu.pth')
print("Training complete and model saved.")