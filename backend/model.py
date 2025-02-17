# analyze_image.py

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from datasets import load_dataset

# Load dataset to get class names
ds = load_dataset("NouRed/plant-disease-recognition")
CLASS_NAMES = ["Healthy", "Powdery", "Rust"]

# Load trained model
model = torch.load("models/plant_disease_model.pth")
# model.eval()  # Set model to evaluation mode


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

    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Predict
    with torch.no_grad():
        outputs = model(image)
        _, class_index = torch.max(outputs, 1)
        class_index = class_index.item()

    return CLASS_NAMES[class_index]


# Example usage
## result = analyze_image("test_leaf.jpg")
## print(f"Detected: {result}")
