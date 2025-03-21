import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import os

# Constants
IMG_SIZE = (224, 224)
DATASET_PATH = "/home/sid/.cache/kagglehub/datasets/vipoooool/new-plant-diseases-dataset/versions/2/train"

# Load dataset
train_dataset = datasets.ImageFolder(root=DATASET_PATH)

# Extract class names dynamically
CLASS_NAMES = list(train_dataset.class_to_idx.keys())
print(f"Updated CLASS_NAMES: {CLASS_NAMES}")


# Define image transformations
def get_transforms():
    return transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# Custom Dataset Class
class PlantDiseaseDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.valid_samples = [
            (img, label) for img, label in dataset if self._is_valid(label)
        ]

        print(f"Number of valid samples: {len(self.valid_samples)}")  # Debugging

    def _is_valid(self, label):
        return 0 <= label < len(CLASS_NAMES)  # Ensure label is within valid range

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        image, label = self.valid_samples[idx]

        # Debugging: Print sample info
        print(f"Index {idx} - Label: {label} ({CLASS_NAMES[label]})")

        if self.transform:
            image = self.transform(image)

        return image, label


# ✅ **Fixed Model - Removed In-Place Operations**
class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes=len(CLASS_NAMES)):
        super(PlantDiseaseModel, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),  # ✅ FIXED: Avoid in-place ReLU
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),  # ✅ FIXED
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),  # ✅ FIXED
            nn.MaxPool2d(2, 2),
        )

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Sequential(
            nn.Linear(128 * 28 * 28, 128),
            nn.ReLU(inplace=False),  # ✅ FIXED
            self.dropout,
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x
