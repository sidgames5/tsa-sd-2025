import tensorflow as tf
import keras
from keras import layers

import numpy as np
import os

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
dataset_path = "dataset"

# Load dataset
train_ds = keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)
val_ds = keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

# Get class names
CLASS_NAMES = train_ds.class_names
print("Classes:", CLASS_NAMES)

# Define CNN model
model = keras.Sequential(
    [
        layers.Rescaling(1.0 / 255, input_shape=(224, 224, 3)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(len(CLASS_NAMES), activation="softmax"),
    ]
)

# Compile and train
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.fit(train_ds, validation_data=val_ds, epochs=10)

# Save model
model.save("models/plant_disease_model.h5")
