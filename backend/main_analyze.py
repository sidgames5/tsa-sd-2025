import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
from backend.model import PlantDiseaseModel
import kagglehub

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 38
BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2

def find_train_folder(root_path):
    """Recursively search for the train folder"""
    for dirpath, dirnames, filenames in os.walk(root_path):
        if 'train' in dirnames:
            return os.path.join(dirpath, 'train')
    return None

def get_data_loaders():
    """Returns train_loader and val_loader with proper validation split"""
    try:
        print("Downloading dataset...")
        dataset_path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
        
        # Search for train folder recursively
        train_path = find_train_folder(dataset_path)
        
        if train_path is None:
            # Alternative approach - check common structures
            possible_paths = [
                os.path.join(dataset_path, "New Plant Diseases Dataset(Augmented)", "train"),
                os.path.join(dataset_path, "new-plant-diseases-dataset", "train"),
                os.path.join(dataset_path, "train")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    train_path = path
                    break
            
            if train_path is None:
                raise FileNotFoundError(f"Could not find train folder in downloaded dataset at: {dataset_path}")

        print(f"Using training data from: {train_path}")

        # Data transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Load dataset
        full_dataset = datasets.ImageFolder(train_path, transform=transform)
        print(f"Found {len(full_dataset)} images in {len(full_dataset.classes)} classes")

        # Split dataset
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
        print(f"Error in get_data_loaders(): {str(e)}")
        print("Please ensure you have:")
        print("1. Kaggle API credentials set up (~/.kaggle/kaggle.json)")
        print("2. Sufficient disk space")
        print("3. Internet connection")
        raise

def train_model():
    """Training workflow"""
    try:
        model = PlantDiseaseModel(num_classes=NUM_CLASSES).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        train_loader, val_loader = get_data_loaders()

        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_acc = correct / total
            epoch_loss /= len(train_loader)
            print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Loss: {epoch_loss:.4f}")

        torch.save(model.state_dict(), "plant_disease_model.pth")
        return model

    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Verify kagglehub is configured
    if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        print("Error: Kaggle API credentials not found. Please:")
        print("1. Go to Kaggle -> Account -> Create API Token")
        print("2. Save kaggle.json to ~/.kaggle/")
        print("3. Run: chmod 600 ~/.kaggle/kaggle.json")
        exit(1)
    
    trained_model = train_model()