import os
import torch
import threading
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
from torchvision import transforms
from backend.model import PlantDiseaseModel, get_transforms
from backend.main_analyze import train_model  # Changed import
from backend.main_analyze import train_metrics

# Flask Setup
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Constants
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]
NUM_CLASSES = 38  # Should match your dataset
UPLOAD_FOLDER = "uploads"
MODEL_PATH = "models/plant_disease_model.pth"
CHART_PATH = "backend/static/accuracy_chart.png"

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Initialize model (global)
model = None
transform = get_transforms()
training_lock = threading.Lock()


def load_model():
    """Load the pre-trained model."""
    global model
    try:
        model = PlantDiseaseModel(num_classes=NUM_CLASSES)
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(
                torch.load(MODEL_PATH, map_location="cpu"),
                strict=True,  # Changed to strict to catch mismatches
            )
        else:
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

        model.eval()
        print("Model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False


# Load model when starting the app
if not load_model():
    print("Warning: Starting without pre-trained model")


def analyze_image(image_path):
    """Perform inference on uploaded plant image."""
    try:
        if model is None:
            raise RuntimeError("Model not loaded")

        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        return CLASS_NAMES[predicted.item()]  # Return class name instead of accuracy
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return None


# main.py


@app.route("/train", methods=["POST"])
def train():
    """Train the model and return metrics"""
    if training_lock.locked():
        return jsonify({"error": "Training is already in progress"}), 409

    with training_lock:
        try:
            print("Training started...")
            metrics = train_model()  # Get metrics from training

            if not load_model():
                raise RuntimeError("Failed to reload model after training")

            return jsonify(
                {
                    "status": "success",
                    "message": "Training complete",
                    "metrics": metrics,
                }
            )
        except Exception as e:
            return jsonify({"error": "Training failed", "details": str(e)}), 500


@app.route("/accuracy/chart", methods=["GET"])
def get_accuracy_chart():
    """Return training metrics"""
    try:
        if os.path.exists("training_metrics.pth"):
            metrics = torch.load("training_metrics.pth")
            return jsonify({"status": True, "data": metrics})
        return (
            jsonify(
                {"error": "Metrics not available", "message": "Train the model first"}
            ),
            404,
        )
    except Exception as e:
        return jsonify({"error": "Server error", "details": str(e)}), 500


if __name__ == "__main__":
    # Preload model before starting server
    if not load_model():
        print("Warning: Starting server without pre-trained model")
    app.run(host="0.0.0.0", port=5000, debug=True)
