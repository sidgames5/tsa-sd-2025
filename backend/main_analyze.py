import os
import torch
import kagglehub
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchvision import datasets, transforms
from backend.model import PlantDiseaseModel, get_transforms

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Download dataset
dataset_path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
if not dataset_path:
    raise FileNotFoundError("Dataset download failed.")

train_path = os.path.join(dataset_path, "train")

# Load dataset
train_dataset = datasets.ImageFolder(root=train_path, transform=get_transforms())
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

CLASS_NAMES = list(train_dataset.class_to_idx.keys())
train_accuracies = []  # Store accuracy per epoch

# Model Setup
model = PlantDiseaseModel(num_classes=len(CLASS_NAMES)).to(device)
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)


def save_accuracy_chart(accuracies):
    """Generates and saves the accuracy graph."""
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(accuracies) + 1),
        accuracies,
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
    os.makedirs("backend/static", exist_ok=True)
    plt.savefig("backend/static/accuracy_chart.png")
    print("Saved accuracy chart at backend/static/accuracy_chart.png")


def train_model():
    """Train the model and return accuracy data."""
    global train_accuracies
    train_accuracies.clear()

    model.train()
    for epoch in range(EPOCHS):
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_accuracies.append(correct / total)
        print(f"Epoch {epoch+1}/{EPOCHS}, Accuracy: {train_accuracies[-1]:.4f}")

    save_accuracy_chart(train_accuracies)
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/plant_disease_model.pth")
    print("Model training complete.")
