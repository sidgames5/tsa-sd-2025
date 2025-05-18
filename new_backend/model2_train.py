import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import os
from datetime import datetime
import numpy as np
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, random_split
import time

# Optimized device configuration
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

# Memory-efficient transforms
def get_transforms():
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

# Lightweight model architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def train_model():
    DATASET_DIR = "new_backend/PlantVillage"
    
    # Check dataset
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Dataset not found at {DATASET_DIR}")
    
    # Load data with progress tracking
    print("Loading dataset...")
    start_time = time.time()
    
    transforms_dict = get_transforms()
    full_dataset = datasets.ImageFolder(DATASET_DIR)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    
    print(f"\nFound {num_classes} classes (1-{num_classes}):")
    for i, name in enumerate(class_names, 1):
        print(f"{i}: {name}")

    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply transforms
    train_dataset.dataset.transform = transforms_dict['train']
    val_dataset.dataset.transform = transforms_dict['val']

    # Optimized data loading
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, 
        num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(
        val_dataset, batch_size=64, 
        num_workers=2, pin_memory=True, persistent_workers=True)

    # Model setup
    model = SimpleCNN(num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, 
        steps_per_epoch=len(train_loader), 
        epochs=20)
    
    criterion = nn.CrossEntropyLoss()

    # Training loop with memory management
    num_epochs = 20
    best_acc = 0.0
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Print stats
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "plant_disease_model.pth")
            print(f"New best model saved with val accuracy: {val_acc:.2f}%")
            
            # Lightweight reporting
            if epoch % 5 == 0:  # Only print full report every 5 epochs
                print("\nClassification Report:")
                print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    print(f"\nTraining completed in {time.time()-start_time:.2f} seconds")
    print(f"Best validation accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    # Memory management wrapper
    torch.backends.cudnn.benchmark = True  # Optimizes CUDA operations
    try:
        train_model()
    except torch.cuda.OutOfMemoryError:
        print("Out of memory! Try reducing batch size or model complexity")
    except KeyboardInterrupt:
        print("Training interrupted - saving current model...")
        torch.save(model.state_dict(), "interrupted_model.pth")