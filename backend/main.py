import cv2
from flask_cors import CORS
import threading
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import torch
from torchvision import transforms
from PIL import Image
from backend.model import PlantDiseaseModel
from backend.main_analyze import train_dataset
from backend.main_analyze import train_model
from backend.main_analyze import train_accuracies

training_lock = threading.Lock()

from backend.main_analyze import save_accuracy_chart


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Enable CORS

CHART_PATH = "backend/static/accuracy_chart.png"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Constants
IMG_SIZE = (224, 224)
CLASS_NAMES = list(train_dataset.class_to_idx.keys())

# Load trained model
model = PlantDiseaseModel(num_classes=len(CLASS_NAMES))
model.load_state_dict(
    torch.load("models/plant_disease_model.pth", map_location=torch.device("cpu"))
)
model.eval()

# Image preprocessing
transform = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def analyze_image(image_path):
    """Perform inference on uploaded plant image."""
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        return CLASS_NAMES[predicted.item()]
    except Exception as e:
        print(f"❌ Error analyzing image: {e}")
        raise e


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Handles image upload and returns prediction."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Invalid file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        result = analyze_image(filepath)
        return jsonify({"message": f"Detected: {result}"})
    except Exception as e:
        return jsonify({"error": "Failed to analyze image", "details": str(e)}), 500


@app.route("/train", methods=["GET"])
def train():
    # Trigger training and return accuracy data.
    global train_accuracies  # modify the global variable
    if training_lock.locked():
        return jsonify({"error": "Training is already in progress"}), 409

    with training_lock:
        try:
            print("🚀 Training started...")
            train_accuracies = train_model()

            if not train_accuracies or not isinstance(train_accuracies, list):
                raise ValueError("Training did not return valid accuracy data.")

            print(f"Training complete. Accuracy: {train_accuracies}")
            return jsonify(
                {"message": "Training complete", "accuracy": train_accuracies}
            )

        except Exception as e:
            print(f"Training Error: {e}")
            return jsonify({"error": "Training failed", "details": str(e)}), 500


@app.route("/accuracy/data", methods=["GET"])
def get_accuracy_data():
    # Return stored training accuracy data.
    if train_accuracies:
        return jsonify({"accuracy": train_accuracies})
    return (
        jsonify({"error": "No accuracy data available yet. Run /api/train first!"}),
        404,
    )


@app.route("/accuracy/chart", methods=["GET"])
def get_accuracy_chart():
    """Return the saved accuracy chart image."""
    if os.path.exists(CHART_PATH):
        return send_file(CHART_PATH, mimetype="image/png")
    return jsonify({"error": "Accuracy chart not found"}), 404


if __name__ == "__main__":
    app.run(debug=True, port=5000)  # Ensure Flask runs on port 5000
