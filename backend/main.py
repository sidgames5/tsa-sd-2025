import os
import cv2
import torch
import threading
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
from torchvision import transforms
from backend.model import PlantDiseaseModel, get_transforms
from backend.main_analyze import train_model, train_accuracies, save_accuracy_chart

# Flask Setup
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Constants
UPLOAD_FOLDER = "uploads"
CHART_PATH = "backend/static/accuracy_chart.png"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load Model
model = PlantDiseaseModel(num_classes=len(train_accuracies))
model.load_state_dict(
    torch.load("models/plant_disease_model.pth", map_location="cpu"), strict=False
)

model.eval()

# Image Transformations
transform = get_transforms()

# Lock for Training
training_lock = threading.Lock()


def analyze_image(image_path):
    """Perform inference on uploaded plant image."""
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        return train_accuracies[predicted.item()]
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return None


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Handles image upload and returns prediction."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Invalid file"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
    file.save(filepath)

    result = analyze_image(filepath)
    if result:
        return jsonify({"message": f"Detected: {result}"})
    return jsonify({"error": "Failed to analyze image"}), 500


@app.route("/api/train", methods=["GET"])
def train():
    """Train the model and return accuracy data."""
    if training_lock.locked():
        return jsonify({"error": "Training is already in progress"}), 409

    with training_lock:
        try:
            train_accuracies.clear()
            print("Training started...")
            train_model()
            save_accuracy_chart(train_accuracies)
            return jsonify(
                {"message": "Training complete", "accuracy": train_accuracies}
            )
        except Exception as e:
            return jsonify({"error": "Training failed", "details": str(e)}), 500


@app.route("/api/accuracy/chart", methods=["GET"])
def get_accuracy_chart():
    # Return the saved accuracy chart image.
    if os.path.exists(CHART_PATH):
        return jsonify({"data": train_accuracies, "status": True})
        # return send_file(CHART_PATH, mimetype="image/png")
    return (
        jsonify(
            {"error": "Internal Server Error", "message": "Accuracy chart not found"}
        ),
        500,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
