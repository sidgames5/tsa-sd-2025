import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm
from backend.model import PlantDiseaseModel
import kagglehub

from collections import defaultdict

train_metrics = defaultdict(list)
# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 38
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2  # 20% of data for validation


def get_data_loaders():
    """Returns train_loader and val_loader with proper validation split"""
    try:
        # Download dataset
        dataset_path = kagglehub.dataset_download(
            "vipoooool/new-plant-diseases-dataset"
        )
        train_path = os.path.join(dataset_path, "train")

        # Data transforms
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # Load full dataset
        full_dataset = datasets.ImageFolder(train_path, transform=transform)

        # Split into train and validation
        val_size = int(VAL_SPLIT * len(full_dataset))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
        )

        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
        )

        return train_loader, val_loader

    except Exception as e:
        print(f"Error creating data loaders: {e}")
        raise


def train_model():
    """Complete training workflow with proper validation"""

    try:
        # Get data loaders
        train_loader, val_loader = get_data_loaders()

        # Initialize model
        model = PlantDiseaseModel(num_classes=NUM_CLASSES).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(EPOCHS):
            model.train()
            epoch_train_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            # Training phase (batch loop)
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Batch metrics
                epoch_train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                epoch_total += labels.size(0)
                epoch_correct += (predicted == labels).sum().item()

            # Epoch-end calculations
            epoch_train_acc = epoch_correct / epoch_total
            epoch_train_loss /= len(train_loader)  # Average loss

            # Store metrics
            train_metrics["accuracies"].append(epoch_train_acc)
            train_metrics["losses"].append(epoch_train_loss)

            print(
                f"Epoch {epoch+1} | Train Acc: {epoch_train_acc:.4f} | Loss: {epoch_train_loss:.4f}"
            )

        return model

    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    model = train_model()
    torch.save(model.state_dict(), "plant_disease_model.pth")
