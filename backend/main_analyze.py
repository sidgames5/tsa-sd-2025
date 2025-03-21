import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import os
from torchvision import datasets, transforms
from backend.model import PlantDiseaseModel
import kagglehub

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 3

# Download dataset
dataset_path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")

# Load dataset
train_path = os.path.join(dataset_path, "train")
valid_path = os.path.join(dataset_path, "valid")

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

train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

CLASS_NAMES = list(train_dataset.class_to_idx.keys())

# Initialize model
model = PlantDiseaseModel(num_classes=len(CLASS_NAMES)).to(device)
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Global list to store accuracy
train_accuracies = []


def save_accuracy_chart(train_accuracies):
    """Generates and saves the accuracy graph."""
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(train_accuracies) + 1),
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
    plt.savefig("backend/static/accuracy_chart.png")
    print("✅ Saved accuracy chart at backend/static/accuracy_chart.png")


def train_model():
    """Train the model and return accuracy data."""
    global train_accuracies
    train_accuracies.clear()  # Reset before training

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

        train_accuracy = correct / total
        train_accuracies.append(train_accuracy)
        print(f"📢 Epoch {epoch+1}/{EPOCHS}, Accuracy: {train_accuracy:.4f}")

    save_accuracy_chart(train_accuracies)
    torch.save(model.state_dict(), "models/plant_disease_model.pth")
    print("🎉 Model training complete.")
    return train_accuracies


if __name__ == "__main__":
    train_model()
