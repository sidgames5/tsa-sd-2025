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
from new_backend.email import send_email
from new_backend.model2_train import SimpleCNN
import pillow_heif

# Enable pillow-heif support
pillow_heif.register_heif_opener()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config.update({
    'MAX_CONTENT_LENGTH': 10 * 1024 * 1024,
    'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg', 'gif', 'heic', 'heif'},
    'MODEL_NAME': "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
    'MODEL_CACHE': "./model_cache",
    'THROTTLE_LIMIT': 5  # requests per minute
})

# Logging setup
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# In-memory history for demo chart
history = {
    "accuracies": [],
    "losses": []
}

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
            config = AutoConfig.from_pretrained(app.config['MODEL_NAME'], trust_remote_code=True, cache_dir=app.config['MODEL_CACHE'])
            self.processor_hf = AutoImageProcessor.from_pretrained(app.config['MODEL_NAME'], trust_remote_code=True, cache_dir=app.config['MODEL_CACHE'])
            self.model_hf = AutoModelForImageClassification.from_pretrained(
                app.config['MODEL_NAME'],
                config=config,
                trust_remote_code=True,
                cache_dir=app.config['MODEL_CACHE']
            ).to(self.device)
            self.labels_hf = self.model_hf.config.id2label
            app.logger.info(f"HuggingFace model loaded.")

            # Load your trained CNN model
            self.model_cnn = SimpleCNN(num_classes=38)
            if os.path.exists("plant_disease_model.pth"):
                self.model_cnn.load_state_dict(torch.load("plant_disease_model.pth", map_location=self.device))
                self.model_cnn = self.model_cnn.to(self.device)
                self.model_cnn.eval()
                app.logger.info(f"SimpleCNN model loaded.")
            else:
                app.logger.error("CNN model file not found! Please train the model first.")
                raise FileNotFoundError("CNN model file not found.")

            # Setup CNN labels from folder names
            dataset_folder = './New Plant Diseases Dataset(Augmented)'
            self.labels_cnn = {idx: class_name for idx, class_name in enumerate(os.listdir(dataset_folder))}

            app.logger.info(f"All models loaded in {time.time() - start_time:.2f} seconds.")

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

            results = []

            # HuggingFace model prediction
            inputs_hf = self.processor_hf(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs_hf = self.model_hf(**inputs_hf)
                logits_hf = outputs_hf.logits
                probs_hf = torch.nn.functional.softmax(logits_hf, dim=-1)
                predicted_idx_hf = logits_hf.argmax(-1).item()
                confidence_hf = probs_hf[0][predicted_idx_hf].item()
                results.append(("HuggingFace", self.labels_hf[predicted_idx_hf], confidence_hf))

            # Your CNN model prediction
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            img_tensor = transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits_cnn = self.model_cnn(img_tensor)
                probs_cnn = torch.nn.functional.softmax(logits_cnn, dim=-1)
                predicted_idx_cnn = logits_cnn.argmax(-1).item()
                confidence_cnn = probs_cnn[0][predicted_idx_cnn].item()
                results.append(("SimpleCNN", self.labels_cnn[predicted_idx_cnn], confidence_cnn))

            # Select model based on plant type
            if plant_type.lower() in ['pepper bell', 'potato', 'tomato', 'tomatoes']:
                selected_model, selected_label, selected_confidence = results[0]
            else:
                selected_model, selected_label, selected_confidence = results[1]

            return {
                "success": True,
                "model_used": selected_model,
                "prediction": selected_label,
                "confidence": round(selected_confidence * 100, 2)
            }

        except Exception as e:
            app.logger.error(f"Prediction error: {str(e)}")
            return {"success": False, "error": str(e)}

# Instantiate classifier
classifier = PlantDiseaseClassifier()

@app.route('/health', methods=['GET'])
def health_check():
    try:
        classifier.load_model()
        return jsonify({
            "status": "healthy",
            "model_loaded": classifier.model_hf is not None and classifier.model_cnn is not None,
            "device": str(classifier.device)
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files or 'plantType' not in request.form:
        return jsonify({"success": False, "error": "Missing image or plantType."}), 400

    image_file = request.files['image']
    plant_type = request.form['plantType']

    if image_file and allowed_file(image_file.filename):
        image_bytes = image_file.read()
        return jsonify(classifier.predict(image_bytes, plant_type))
    else:
        return jsonify({"success": False, "error": "Invalid file type."}), 400

@app.route("/accuracy/chart", methods=["GET"])
def get_chart_data():
    return jsonify({
        "success": True,
        "data": {
            "accuracies": history["accuracies"],
            "losses": history["losses"]
        }
    })

@app.route("/send-results", methods=['POST'])
def send_results():
    data = request.get_json()
    email = data.get('email')
    results = data.get('results', [])
    good_results = []
    for result in results:
        good_results.append({
            "name": result["name"],
            "confidence": result["confidence"],
            "prediction": result["status"]
        })
    message = data.get('message', f"""<html><body><h1>Your LeafLogic report is ready!</h1><table><tr><th>Name</th><th>Confidence</th><th>Status</th></tr>{''.join(f"<tr><td>{res['name']}</td><td>{res['confidence']}</td><td>{res['prediction']}</td></tr>" for res in good_results)}</table><p>Thank you for using LeafLogic!</p><p>Best regards,<br>LeafLogic Team</p></body></html>""")

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