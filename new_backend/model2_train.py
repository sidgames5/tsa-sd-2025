
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import os

# Check for GPU availability with MPS support
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Dataset directory path
DATASET_DIR = "/Users/kaniskprakash/Documents/GitHub/tsa-sd-2025/new_backend/New Plant Diseases Dataset(Augmented)"

# Dataset transformations: resizing to 224x224 and normalizing images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224
    transforms.ToTensor(),  # Convert images to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization (standard for pre-trained models)
])

# Check if dataset directory exists
if not os.path.exists(DATASET_DIR):
    print(f"Dataset directory {DATASET_DIR} not found. Please check the path.")
else:
    # Load dataset using ImageFolder
    full_dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)

    # Split dataset into training and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # DataLoaders for batching
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Define a custom CNN model (training from scratch)
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(128 * 28 * 28, 512)  # Adjust size for the output of conv layers
            self.fc2 = nn.Linear(512, num_classes)
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.pool(torch.relu(self.conv3(x)))
            x = x.view(-1, 128 * 28 * 28)  # Flatten the output of conv layers
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)  # Apply dropout
            x = self.fc2(x)
            return x

    # Initialize model, loss function, and optimizer
    num_classes = len(full_dataset.classes)  # Get the number of unique classes from the dataset
    model = SimpleCNN(num_classes=num_classes).to(device)  # Move model to the device

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f"\nEpoch [{epoch+1}/{num_epochs}] - Training")

        # Training phase
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print training loss for each batch
            if batch_idx % 10 == 0:  # Print every 10th batch to avoid too much output
                print(f"  Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")

        # Validation phase after each epoch
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

    # Save the trained model
    torch.save(model.state_dict(), "plant_disease_model.pth")
    print("\nModel training complete and saved!")