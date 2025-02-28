import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
import numpy as np
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 8
CLASS_NAMES = ["Healthy", "Powdery", "Rust"]

# Load dataset from Hugging Face
ds = load_dataset("NouRed/plant-disease-recognition")
print(ds["train"][0])  # Debugging

# Define image transformations
transform = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),  # Resize to 224x224
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.RandomRotation(10),  # Data augmentation
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2
        ),  # Data augmentation
        transforms.ToTensor(),  # Convert PIL image to tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize
    ]
)


# Custom Dataset Class
class PlantDiseaseDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.valid_indices = [
            i for i, sample in enumerate(dataset) if self._is_valid(sample)
        ]
        print(f"Number of valid samples: {len(self.valid_indices)}")  # Debugging

    def _is_valid(self, sample):
        if isinstance(sample, dict):
            image = sample.get("image")
            label_text = sample.get("text")

            # Check if image is valid
            if image is None:
                print(f"Invalid sample: Image is None")
                return False

            # Check if label_text is valid
            if label_text is None:
                print(f"Invalid sample: Label is None")
                return False

            # Strip whitespace and check case-insensitive
            label_text = label_text.strip().capitalize()
            if label_text not in [c.capitalize() for c in CLASS_NAMES]:
                print(f"Invalid sample: Label '{label_text}' not in CLASS_NAMES")
                return False

            return True

        print(f"Invalid sample: Not a dictionary")
        return False

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        example = self.dataset[actual_idx]

        image = example["image"]  # PIL image
        label_text = example["text"]  # Label text

        # Debugging: Print image and label info
        print(f"Index {idx} - Image shape: {image.size}, Label: {label_text}")

        # Convert label text to index
        label = CLASS_NAMES.index(label_text.strip().capitalize())

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label


# Prepare training and validation datasets
from torch.utils.data import random_split

# Split the train dataset into 80% train and 20% validation
train_size = int(0.8 * len(ds["train"]))
val_size = len(ds["train"]) - train_size

train_data, val_data = random_split(ds["train"], [train_size, val_size])

# Create Dataset and DataLoader
train_dataset = PlantDiseaseDataset(train_data, transform=transform)
val_dataset = PlantDiseaseDataset(val_data, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Bootstrapping: Oversample "Healthy" class
def bootstrap_data(dataset, target_class="Healthy", multiplier=3):
    # Oversample images from the specified class to balance dataset.
    new_data = []
    for idx in range(len(dataset)):  # Use indices to access dataset items
        sample = dataset[idx]  # Get sample using __getitem__
        if sample is None:
            continue
        image, label = sample
        if label == CLASS_NAMES.index(target_class):
            new_data.extend([(image, label)] * multiplier)  # Duplicate the data
        else:
            new_data.append((image, label))  # Keep other classes the same
    return new_data


# Apply bootstrapping to "Healthy" class only (for testing)
bootstrapped_train_data = bootstrap_data(
    train_dataset, target_class="Healthy", multiplier=4
)

train_dataset = PlantDiseaseDataset(bootstrapped_train_data, transform=transform)
val_dataset = PlantDiseaseDataset(val_data, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Define CNN model with Dropout Regularization
class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes=len(CLASS_NAMES)):
        super(PlantDiseaseModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)  # Helps prevent overfitting
        self.fc1 = nn.Linear(128 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(nn.ReLU()(self.fc1(x)))  # Dropout applied here
        x = self.fc2(x)
        return x


# Instantiate model, loss function, and optimizer
model = PlantDiseaseModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
def train_model():
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader)}")


# Validation loop
def validate_model():
    model.eval()  # Set model to evaluation mode
    correct = 0  # Counter for correct predictions
    total = 0  # Counter for total predictions
    with torch.no_grad():  # Disable gradient calculation
        for images, labels in val_loader:  # Iterate over validation batches
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # Forward pass
            _, predicted = torch.max(outputs, 1)  # Get the index of the highest score
            total += labels.size(0)  # Total number of labels
            correct += (predicted == labels).sum().item()  # Count correct predictions

    # Calculate and print accuracy
    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy}%")


# Train and validate
train_model()
validate_model()

# Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/plant_disease_model.pth")
print("Model saved as plant_disease_model.pth")
