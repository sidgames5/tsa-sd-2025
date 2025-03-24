import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import Dataset

# Constants
IMG_SIZE = (224, 224)
DATASET_PATH = "/home/sid/.cache/kagglehub/datasets/vipoooool/new-plant-diseases-dataset/versions/2/train"

# Load dataset
train_dataset = datasets.ImageFolder(root=DATASET_PATH)
CLASS_NAMES = list(train_dataset.class_to_idx.keys())


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


class PlantDiseaseDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.valid_samples = [
            (img, label) for img, label in dataset if self._is_valid(label)
        ]

    def _is_valid(self, label):
        return 0 <= label < len(CLASS_NAMES)

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        image, label = self.valid_samples[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseModel, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Updated to match saved model
        self.fc = nn.Sequential(
            nn.Linear(128 * 28 * 28, 128),  # Use 128 instead of 512
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
