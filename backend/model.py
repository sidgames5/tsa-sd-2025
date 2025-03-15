import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
from datasets import load_dataset

# Constants
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Healthy", "Powdery", "Rust"]

# Define image transformations
def get_transforms():
    return transforms.Compose(
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
