import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import os
from torchvision import datasets, transforms
from backend.model import PlantDiseaseModel
import kagglehub

# Device configuration
device = torch.device("cuda")

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Download dataset
dataset_path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")

# Verify dataset paths
train_path = os.path.join(dataset_path, "train")
valid_path = os.path.join(dataset_path, "valid")
print("Checking dataset paths:")
print(f"Train Path Exists: {os.path.exists(train_path)}")
print(f"Valid Path Exists: {os.path.exists(valid_path)}")

# Image transformations
transform = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load dataset
train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
val_dataset = datasets.ImageFolder(root=valid_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

CLASS_NAMES = list(train_dataset.class_to_idx.keys())
print(f"Number of Classes: {len(CLASS_NAMES)}")
print(f"Updated CLASS_NAMES: {CLASS_NAMES}")

# Initialize model
model = PlantDiseaseModel(num_classes=len(CLASS_NAMES)).to(device)
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)


# Training
def train_model():
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Print batch labels only for the first batch of the first epoch
            if epoch == 0 and i == 0:
                unique_labels = labels.unique().tolist()
                print(f"Batch Labels (first batch only): {unique_labels}")

            if any(l >= len(CLASS_NAMES) or l < 0 for l in labels.tolist()):
                raise ValueError(f"Invalid label detected: {labels.tolist()}")

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Limit batch processing for debugging (remove in production)
            if i > 10:
                break

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss / len(train_loader):.4f}")


train_model()
torch.save(model.state_dict(), "models/plant_disease_model.pth")
print("Model training complete.")
