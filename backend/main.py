import os
import torch
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
from torchvision import transforms
from backend.model import PlantDiseaseModel, get_transforms
from backend.main_analyze import train_model, CLASS_NAMES  # Changed import

# Flask Setup
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Constants
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
                strict=True  # Changed to strict to catch mismatches
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
        if result:
            return jsonify({
                "status": "success",
                "prediction": result,
                "confidence": "high"  # You can add confidence scoring later
            })
        return jsonify({"error": "Analysis failed"}), 500
    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route("/api/train", methods=["POST"])  # Changed to POST
def train():
    """Train the model (admin-only endpoint)."""
    if training_lock.locked():
        return jsonify({"error": "Training is already in progress"}), 409

    # Add authentication check here if needed
    # if not request.headers.get("Authorization") == "YOUR_SECRET_KEY":
    #     return jsonify({"error": "Unauthorized"}), 401

    with training_lock:
        try:
            print("Training started...")
            train_model()  # This saves the new model automatically
            
            # Reload the updated model
            if not load_model():
                raise RuntimeError("Failed to reload model after training")
                
            return jsonify({
                "status": "success",
                "message": "Training complete",
                "accuracy": "See /api/accuracy/chart for details"
            })
        except Exception as e:
            return jsonify({
                "error": "Training failed",
                "details": str(e)
            }), 500

@app.route("/api/accuracy/chart", methods=["GET"])
def get_accuracy_chart():
    """Get training accuracy data."""
    try:
        if os.path.exists(CHART_PATH):
            return send_file(
                CHART_PATH,
                mimetype="image/png",
                as_attachment=False
            )
        return jsonify({
            "error": "Chart not available",
            "message": "Train the model first"
        }), 404
    except Exception as e:
        return jsonify({
            "error": "Server error",
            "details": str(e)
        }), 500

@app.route("/api/health", methods=["GET"])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        "status": "ready" if model else "no_model",
        "model_loaded": model is not None
    })

if __name__ == "__main__":
    # Preload model before starting server
    if not load_model():
        print("Warning: Starting server without pre-trained model")
    app.run(host="0.0.0.0", port=5000, debug=True)