import torch
import torch.nn as nn
from torchvision import transforms


class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()
        
        self.features = nn.Sequential(
            # Input: 3x224x224
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x112x112
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x56x56
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128x28x28
        )
        
        # Automatic flattened size calculation
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            self.flattened_size = self.features(dummy).numel()
        
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        print(f"Feature map shape: {x.shape}")  # Debug print
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def get_transforms():
    return transforms.Compose([
    transforms.Resize((224, 224)),  # Must be exact
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                            [0.229, 0.224, 0.225])
])