import os
import io
import time
import torch
import logging
from PIL import Image
from flask import Flask, request, jsonify, flask_email
from flask_cors import CORS
from logging.handlers import RotatingFileHandler
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification
from werkzeug.utils import secure_filename
from new_backend.email import send_email

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config.update({
    'MAX_CONTENT_LENGTH': 10 * 1024 * 1024,
    'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg', 'gif'},
    'MODEL_NAME': "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
    'MODEL_CACHE': "./model_cache",
    'THROTTLE_LIMIT': 5  # requests per minute
})

# Logging setup
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# In-memory history for demo chart (you can replace this with a real DB)
history = {
    "accuracies": [],
    "losses": []
}

class PlantDiseaseClassifier:
    def __init__(self):
        self.model = None
        self.processor = None
        self.labels = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_request_time = 0
        app.logger.info(f"Using device: {self.device}")

    def load_model(self):
        if self.model is not None:
            return
        try:
            start_time = time.time()
            app.logger.info("Loading model...")
            config = AutoConfig.from_pretrained(app.config['MODEL_NAME'], trust_remote_code=True, cache_dir=app.config['MODEL_CACHE'])
            self.processor = AutoImageProcessor.from_pretrained(app.config['MODEL_NAME'], trust_remote_code=True, cache_dir=app.config['MODEL_CACHE'])
            self.model = AutoModelForImageClassification.from_pretrained(
                app.config['MODEL_NAME'],
                config=config,
                trust_remote_code=True,
                cache_dir=app.config['MODEL_CACHE']
            ).to(self.device)
            self.labels = self.model.config.id2label
            app.logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds with labels: {list(self.labels.values())}")
        except Exception as e:
            app.logger.error(f"Model loading failed: {str(e)}")
            raise

    def preprocess_image(self, image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes))
            return image.convert('RGB') if image.mode != 'RGB' else image
        except Exception as e:
            app.logger.error(f"Image processing failed: {str(e)}")
            raise ValueError("Invalid image file")

    def predict(self, image_bytes):
        try:
            current_time = time.time()
            if current_time - self.last_request_time < 60 / app.config['THROTTLE_LIMIT']:
                time.sleep(60 / app.config['THROTTLE_LIMIT'] - (current_time - self.last_request_time))
            self.last_request_time = time.time()

            self.load_model()

            image = self.preprocess_image(image_bytes)
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                start_time = time.time()
                outputs = self.model(**inputs)
                inference_time = time.time() - start_time

                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                predicted_idx = logits.argmax(-1).item()
                confidence = probs[0][predicted_idx].item()

                # Store fake chart data
                history["accuracies"].append(round(confidence, 4))
                history["losses"].append(round(1 - confidence, 4))

                return {
                    "success": True,
                    "prediction": self.labels[predicted_idx],
                    "confidence": round(confidence * 100, 2),
                    "inference_time": round(inference_time, 2)
                }

        except Exception as e:
            app.logger.error(f"Prediction error: {str(e)}")
            return {"success": False, "error": str(e)}

# Instantiate classifier
classifier = PlantDiseaseClassifier()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/health', methods=['GET'])
def health_check():
    try:
        classifier.load_model()
        return jsonify({
            "status": "healthy",
            "model_loaded": classifier.model is not None,
            "device": str(classifier.device)
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image provided"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid file type"}), 400

    try:
        image_bytes = file.read()
        if len(image_bytes) == 0:
            raise ValueError("Empty file content")

        result = classifier.predict(image_bytes)
        return jsonify(result), 200 if result["success"] else 500

    except Exception as e:
        app.logger.error(f"Server error: {str(e)}")
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500

@app.route("/accuracy/chart", methods=["GET"])
def get_chart_data():
    return jsonify({
        "success": True,
        "data": {
            "accuracies": history["accuracies"],
            "losses": history["losses"]
        }
    })

@app.route("/send-results",methods=['POST'])
def send_results():
    data = request.get_json()
    email = data.get('email')
    message = data.get('message', 'Here are your latest LeafLogic scan results.')

    if not email:
        return jsonify({"success": False, "error": "Email is required"}), 400

    success = send_email(email, message)
    if success:
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "error": "Failed to send email"}), 500
    

if __name__ == "__main__":
    try:
        classifier.load_model()
    except Exception as e:
        app.logger.critical(f"Failed to load model: {str(e)}")

    app.run(host="0.0.0.0", port=5000, debug=False)
