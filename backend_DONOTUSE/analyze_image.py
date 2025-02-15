import tensorflow as tf
import numpy as np
import cv2
import os

# Load trained model
model = tf.keras.models.load_model("models/plant_disease_model.h5")
CLASS_NAMES = np.load("models/class_names.npy")

CLASS_NAMES = ["Healthy", "Diseased"]  # Update based on dataset


def analyze_image(image_path):
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0) / 255.0

    # Predict
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    return CLASS_NAMES[class_index]
