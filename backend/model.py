# analyze_image.py

import keras
import tensorflow as tf
import numpy as np
import cv2
from datasets import load_dataset

# Load dataset to get class names
ds = load_dataset("NouRed/plant-disease-recognition")
CLASS_NAMES = ["Healthy", "Powdery", "Rust"]

# Load trained model
model = keras.models.load_model("models/plant_disease_model.h5")


def analyze_image(image_path):
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0) / 255.0

    # Predict
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    return CLASS_NAMES[class_index]


# Example usage
result = analyze_image("test_leaf.jpg")
print(f"Detected: {result}")
