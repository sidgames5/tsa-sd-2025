import torch
import matplotlib as plt

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
EPOCHS = 3

# Download dataset
dataset_path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")

# Verify dataset paths
train_path = os.path.join(dataset_path, "train")
valid_path = os.path.join(dataset_path, "valid")
# print("Checking dataset paths:")
# print(f"Train Path Exists: {os.path.exists(train_path)}")
# print(f"Valid Path Exists: {os.path.exists(valid_path)}")

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
# rint(f"Number of Classes: {len(CLASS_NAMES)}")
# print(f"Updated CLASS_NAMES: {CLASS_NAMES}")

# Initialize model
model = PlantDiseaseModel(num_classes=len(CLASS_NAMES)).to(device)
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

train_accuracies = []
val_accuracies = []


# Training
def train_model():
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0  # Initialize correct predictions counter
        total = 0  # Initialize total samples counter

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

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

            # Compute Accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_accuracy = correct / total  # Calculate accuracy
        train_accuracies.append(train_accuracy)

        print(
            f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {train_accuracy:.4f}"
        )

    # Generate and save the accuracy graph

    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, EPOCHS + 1),
        train_accuracies,
        marker="o",
        linestyle="-",
        label="Training Accuracy",
        color="blue",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("backend/static/accuracy_chart.png")  # Save the graph
    print("Saved accuracy chart at backend/static/accuracy_chart.png")


train_model()
torch.save(model.state_dict(), "models/plant_disease_model.pth")
print("Model training complete.")
