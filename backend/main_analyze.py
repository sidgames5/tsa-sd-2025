import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torchvision import datasets
from backend.nmodel import PlantDiseaseModel
from backend.model import get_transforms

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
NUM_CLASSES = 38  # New Plant Diseases Dataset classes
DATA_PATH = "data/plant_diseases"  # Update with your path

# Global tracking
CLASS_NAMES = []
train_history = {'loss': [], 'accuracy': []}

def load_datasets():
    """Load and split datasets with augmentation"""
    global CLASS_NAMES
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    full_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_PATH, "train"),
        transform=train_transform
    )
    
    CLASS_NAMES = full_dataset.classes
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )
    
    return (
        DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_dataset, batch_size=BATCH_SIZE)
    )

def train_model():
    """Complete training workflow"""
    train_loader, val_loader = load_datasets()
    
    model = PlantDiseaseModel(num_classes=len(CLASS_NAMES)).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = CrossEntropyLoss()
    
    best_accuracy = 0.0
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation phase
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        
        # Track metrics
        train_history['loss'].append(val_loss)
        train_history['accuracy'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), "models/plant_disease_model.pth")
            print(f"Saved new best model with accuracy: {best_accuracy:.4f}")
    
    save_training_plots()
    return best_accuracy

def evaluate(model, dataloader, criterion):
    """Evaluate model on validation set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return running_loss / len(dataloader), correct / total

def save_training_plots():
    """Save training metrics visualization"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_history['loss'], label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_history['accuracy'], label='Accuracy', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    os.makedirs("backend/static", exist_ok=True)
    plt.savefig("backend/static/training_metrics.png")
    plt.close()