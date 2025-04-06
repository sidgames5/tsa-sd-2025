import os
import io
import time
import smtplib
import torch
import logging
from email.message import EmailMessage
from dotenv import load_dotenv
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForImageClassification, AutoImageProcessor, AutoConfig
from logging.handlers import RotatingFileHandler

# Load environment variables
load_dotenv()

# Configuration
MODEL_NAME = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
MODEL_CACHE = "./model_cache"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
THROTTLE_LIMIT = 5  # requests per minute

# Email credentials
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")

# App setup
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Logger setup
log_handler = RotatingFileHandler("app.log", maxBytes=10000, backupCount=3)
log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
app.logger.addHandler(log_handler)
app.logger.setLevel(logging.INFO)

# Fake chart data
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

    def load_model(self):
        if self.model:
            return
        config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True, cache_dir=MODEL_CACHE)
        self.processor = AutoImageProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True, cache_dir=MODEL_CACHE)
        self.model = AutoModelForImageClassification.from_pretrained(
            MODEL_NAME, config=config, trust_remote_code=True, cache_dir=MODEL_CACHE
        ).to(self.device)
        self.labels = self.model.config.id2label

    def preprocess_image(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes))
        return image.convert("RGB")

    def predict(self, image_bytes):
        current_time = time.time()
        if current_time - self.last_request_time < 60 / THROTTLE_LIMIT:
            time.sleep(60 / THROTTLE_LIMIT - (current_time - self.last_request_time))
        self.last_request_time = time.time()

        self.load_model()
        image = self.preprocess_image(image_bytes)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            start = time.time()
            outputs = self.model(**inputs)
            inference_time = time.time() - start

            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx = logits.argmax(-1).item()
            confidence = probs[0][idx].item()

            history["accuracies"].append(round(confidence, 4))
            history["losses"].append(round(1 - confidence, 4))

            return {
                "success": True,
                "prediction": self.labels[idx],
                "confidence": round(confidence * 100, 2),
                "inference_time": round(inference_time, 2)
            }


classifier = PlantDiseaseClassifier()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def send_email_result(to_email, result):
    if not EMAIL_USER or not EMAIL_PASS:
        raise Exception("Email credentials not set in .env")

    msg = EmailMessage()
    msg["Subject"] = "Your Plant Disease Detection Result 🌿"
    msg["From"] = EMAIL_USER
    msg["To"] = to_email

    content = f"""\
✅ Prediction: {result['prediction']}
📊 Confidence: {result['confidence']}%
⚡ Inference Time: {result['inference_time']}s

Thanks for using our app!
"""
    msg.set_content(content)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL_USER, EMAIL_PASS)
        smtp.send_message(msg)


@app.route("/health", methods=["GET"])
def health_check():
    try:
        classifier.load_model()
        return jsonify({"status": "healthy", "device": str(classifier.device)})
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "Image file is required"}), 400

    image_file = request.files["image"]
    email = request.form.get("email")

    if image_file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    if not allowed_file(image_file.filename):
        return jsonify({"success": False, "error": "Invalid file type"}), 400

    try:
        image_bytes = image_file.read()
        result = classifier.predict(image_bytes)

        if result["success"] and email:
            send_email_result(email, result)

        return jsonify(result), 200
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
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


if __name__ == "__main__":
    try:
        classifier.load_model()
        app.logger.info("Model loaded successfully")
    except Exception as e:
        app.logger.critical(f"Startup failed: {str(e)}")
    app.run(host="0.0.0.0", port=5000, debug=False)
