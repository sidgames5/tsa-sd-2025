import os
import io
import time
import torch
import logging
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from logging.handlers import RotatingFileHandler
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification
from torchvision import transforms
from werkzeug.utils import secure_filename
from new_backend.model2_train import SimpleCNN
import pillow_heif
import ollama
import re
import json
from functools import wraps
from dotenv import load_dotenv

# Initialize environment
load_dotenv()
REVIEW_FILE = "new_backend/reviews.json"
pillow_heif.register_heif_opener()

# Flask app setup
app = Flask(__name__)
CORS(app)

# Configuration
app.config.update({
    'MAX_CONTENT_LENGTH': 10 * 1024 * 1024,
    'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg', 'gif', 'heic', 'heif'},
    'MODEL_NAME': "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
    'MODEL_CACHE': "./model_cache",
    'THROTTLE_LIMIT': 5,
    'DATASET_DIR': "new_backend/PlantVillage"
})

# Admin credentials
ADMIN_CREDENTIALS = {
    "username": os.getenv("ADMIN_USERNAME", "admin"),
    "password": os.getenv("ADMIN_PASSWORD", "leafadmin123")
}

# Setup directories
os.makedirs('static/reviews', exist_ok=True)
if not os.path.exists('static/reviews/default_user.png'):
    img = Image.new('RGB', (200, 200), color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    draw.text((50, 80), "User", fill=(0, 0, 0))
    img.save('static/reviews/default_user.png')

# Logging configuration
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# Initialize history
history = {"accuracies": [], "losses": []}

class PlantDiseaseClassifier:
    def __init__(self):
        self.model_hf = None
        self.model_cnn = None
        self.processor_hf = None
        self.labels_hf = None
        self.labels_cnn = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        app.logger.info(f"Using device: {self.device}")

    def load_model(self):
        if self.model_hf is not None and self.model_cnn is not None:
            return

        try:
            start_time = time.time()
            app.logger.info("Loading models...")

            # Load HuggingFace model
            config = AutoConfig.from_pretrained(
                app.config['MODEL_NAME'],
                trust_remote_code=True,
                cache_dir=app.config['MODEL_CACHE']
            )
            self.processor_hf = AutoImageProcessor.from_pretrained(
                app.config['MODEL_NAME'],
                trust_remote_code=True,
                cache_dir=app.config['MODEL_CACHE']
            )
            self.model_hf = AutoModelForImageClassification.from_pretrained(
                app.config['MODEL_NAME'],
                config=config,
                trust_remote_code=True,
                cache_dir=app.config['MODEL_CACHE']
            ).to(self.device)
            self.labels_hf = self.model_hf.config.id2label
            app.logger.info("HuggingFace model loaded.")

            # Load CNN model
            dataset_folder = app.config['DATASET_DIR']
            if not os.path.exists(dataset_folder):
                raise FileNotFoundError(f"Dataset folder not found: {dataset_folder}")

            class_names = [d for d in os.listdir(dataset_folder) 
                         if os.path.isdir(os.path.join(dataset_folder, d))]
            num_classes = len(class_names)
            
            self.model_cnn = SimpleCNN(num_classes=num_classes).to(self.device)
            if os.path.exists("plant_disease_model.pth"):
                self.model_cnn.load_state_dict(
                    torch.load("plant_disease_model.pth", map_location=self.device)
                )
                self.model_cnn.eval()
                app.logger.info("SimpleCNN model loaded.")
            else:
                raise FileNotFoundError("CNN model file not found.")

            self.labels_cnn = {idx: name for idx, name in enumerate(class_names)}
            app.logger.info(f"Loaded {num_classes} classes for CNN model")

            app.logger.info(f"Models loaded in {time.time()-start_time:.2f}s")

        except Exception as e:
            app.logger.error(f"Model loading failed: {str(e)}")
            raise

    def preprocess_image(self, image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes))
            return image.convert('RGB') if image.mode != 'RGB' else image
        except Exception as e:
            app.logger.error(f"Image processing failed: {str(e)}")
            raise ValueError("Invalid or unsupported image file")

    def predict(self, image_bytes, plant_type):
        try:
            self.load_model()
            image = self.preprocess_image(image_bytes)

            use_cnn_only = plant_type.strip().lower() in ['pepper bell', 'potato', 'tomato']
            results = []

            if not use_cnn_only:
                try:
                    inputs = self.processor_hf(images=image, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = self.model_hf(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        pred_idx = outputs.logits.argmax(-1).item()
                        confidence = int(probs[0][pred_idx].item() * 100)
                        label = self.labels_hf.get(pred_idx, "Unknown")
                        results.append(("HuggingFace", label, confidence))
                except Exception as e:
                    app.logger.warning(f"HF prediction failed: {e}")

            if use_cnn_only or not results:
                try:
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor()
                    ])
                    img_tensor = transform(image).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        outputs = self.model_cnn(img_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=-1)
                        pred_idx = outputs.argmax(-1).item()
                        confidence = int(probs[0][pred_idx].item() * 100)
                        label = self.labels_cnn.get(pred_idx, "Unknown")
                        results.append(("SimpleCNN", label, confidence))
                except Exception as e:
                    app.logger.warning(f"CNN prediction failed: {e}")

            if not results:
                raise ValueError("No model produced a prediction")

            model_used, prediction, confidence = results[-1]
            return {
                "success": True,
                "model_used": model_used,
                "prediction": prediction,
                "confidence": confidence
            }

        except Exception as e:
            app.logger.error(f"Prediction error: {str(e)}")
            return {"success": False, "error": str(e)}

# Initialize classifier
classifier = PlantDiseaseClassifier()

# Helper functions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def ensure_reviews_file():
    if not os.path.exists(REVIEW_FILE):
        with open(REVIEW_FILE, 'w') as f:
            json.dump([], f)
    else:
        try:
            with open(REVIEW_FILE, 'r') as f:
                json.load(f)
        except json.JSONDecodeError:
            with open(REVIEW_FILE, 'w') as f:
                json.dump([], f)

ensure_reviews_file()

# Routes
@app.route('/health', methods=['GET'])
def health_check():
    try:
        classifier.load_model()
        return jsonify({
            "status": "healthy",
            "models_loaded": classifier.model_hf is not None and classifier.model_cnn is not None
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files or 'plantType' not in request.form:
        return jsonify({"success": False, "error": "Missing image or plantType"}), 400

    file = request.files['image']
    plant_type = request.form['plantType']

    if file and allowed_file(file.filename):
        try:
            image_bytes = file.read()
            return jsonify(classifier.predict(image_bytes, plant_type))
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
    return jsonify({"success": False, "error": "Invalid file type"}), 400

# ... (keep all other routes the same as in your original file)

if __name__ == "__main__":
    try:
        classifier.load_model()
        app.run(host="0.0.0.0", port=5000, debug=False)
    except Exception as e:
        app.logger.critical(f"Failed to start: {str(e)}")