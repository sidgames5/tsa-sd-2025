import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from datasets import load_dataset

# Load dataset to get class names
CLASS_NAMES = ["Healthy", "Powdery", "Rust"]


# Define the same model architecture as used during training
class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes=3):  # Adjust `num_classes` to match your dataset
        super(PlantDiseaseModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 128)  # Match dimensions from training
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x


# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PlantDiseaseModel(
    num_classes=len(CLASS_NAMES)
)  # Use 3 classes (Healthy, Powdery, Rust)
state_dict = torch.load(
    "models/plant_disease_model.pth", map_location=device
)  # Load state dict
model.load_state_dict(state_dict)  # Load weights into model
model.to(device)  # Move model to device (CPU or GPU)
model.eval()  # Set to evaluation mode

# Image preprocessing pipeline
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def analyze_image(image_path):
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        raise ValueError(f"Image not found at path: {image_path}")

    image = (
        transform(image).unsqueeze(0).to(device)
    )  # Add batch dimension and move to device

    # Predict
    with torch.no_grad():
        outputs = model(image)
        _, class_index = torch.max(outputs, 1)
        class_index = class_index.item()

    return CLASS_NAMES[class_index]


# Example usage
# result = analyze_image("test_leaf.jpg")
# print(f"Detected: {result}")
