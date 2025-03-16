import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import os
from torchvision import datasets, transforms
from train.model import PlantDiseaseModel
import kagglehub

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 8
CLASS_NAMES = ["Healthy", "Powdery", "Rust"]  # Update based on dataset classes

# Download the dataset using kagglehub
print("Downloading dataset...")
dataset_path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
print(f"Dataset downloaded to: {dataset_path}")

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

# Load dataset from the downloaded path
train_dataset = datasets.ImageFolder(
    root=os.path.join(dataset_path, "train"), transform=transform
)
val_dataset = datasets.ImageFolder(
    root=os.path.join(dataset_path, "valid"), transform=transform
)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Bootstrapping: Oversample "Healthy" class
def bootstrap_data(dataset, target_class="Healthy", multiplier=3):
    # Oversample images from the specified class to balance dataset.
    new_data = []
    for idx in range(len(dataset)):
        image, label = dataset[idx]
        if CLASS_NAMES[label] == target_class:
            new_data.extend([(image, label)] * multiplier)  # Duplicate the data
        else:
            new_data.append((image, label))  # Keep other classes the same
    return new_data


# Apply bootstrapping to "Healthy" class only (for testing)
bootstrapped_train_data = bootstrap_data(
    train_dataset, target_class="Healthy", multiplier=4
)


# Create a custom dataset from bootstrapped data
class BootstrappedDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


train_dataset = BootstrappedDataset(bootstrapped_train_data)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Instantiate model, loss function, and optimizer
model = PlantDiseaseModel(num_classes=len(CLASS_NAMES)).to(device)
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)


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
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy}%")


# Train and validate
train_model()
validate_model()

# Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/plant_disease_model.pth")
print("Model saved as plant_disease_model.pth")
