import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
import numpy as np
import cv2
import os
from torch.utils.data import random_split

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
CLASS_NAMES = ["Healthy", "Powdery", "Rust"]

# Load dataset from Hugging Face
ds = load_dataset("NouRed/plant-disease-recognition")

# Define image transformations
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Custom Dataset Class
class PlantDiseaseDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = np.array(example["image"])
        label = CLASS_NAMES.index(example["text"])

        # Preprocess image
        if self.transform:
            image = self.transform(image)

        return image, label


# Prepare training and validation datasets

from torch.utils.data import random_split

# Split the train dataset into 80% train and 20% validation
train_size = int(0.8 * len(ds["train"]))
val_size = len(ds["train"]) - train_size

train_data, val_data = random_split(ds["train"], [train_size, val_size])

train_dataset = PlantDiseaseDataset(train_data, transform=transform)
val_dataset = PlantDiseaseDataset(val_data, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Define CNN model
class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes=len(CLASS_NAMES)):
        super(PlantDiseaseModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
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
