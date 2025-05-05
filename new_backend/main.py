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
from new_backend.email import send_email
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
    good_results = [{
        "name": r["name"],
        "confidence": r["confidence"],
        "prediction": r["status"]
    } for r in results]

    message = data.get('message', f"""<html><body><h1>Your LeafLogic report is ready!</h1><table><tr><th>Name</th><th>Confidence</th><th>Status</th></tr>{''.join(f"<tr><td>{res['name']}</td><td>{res['confidence']}</td><td>{res['prediction']}</td></tr>" for res in good_results)}</table><p>Thank you for using LeafLogic!</p><p>Best regards,<br>LeafLogic Team</p></body></html>""")

    if not email:
        return jsonify({"success": False, "error": "Email is required"}), 400

    success = send_email(email, message)
    return jsonify({"success": success, "error": None if success else "Failed to send email"}), 500 if not success else 200

@app.route('/ollama-support', methods=['POST'])
def ollama_support():
    data = request.get_json()
    prompt = data.get("prompt", "").strip().lower()

    # Static responses for certain steps like uploading images or feature-related questions
    feature_keywords = [
        "features", "what can you do", "capabilities", "app features",
        "how can you help", "what's good", "functions", "tools",
        "what is leaflogic", "tell me about leaflogic", "leaflogic features"
    ]
    
    upload_keywords = [
        "upload", "how to upload", "upload an image", "how do I upload"
    ]
    
    # Fixed feature list response
    hardcoded_feature_reply = """
Sure! Here are the features of LeafLogic: \n

- 📸 **Plant Disease Detection**: Upload a photo of a plant to detect diseases using advanced AI. \n
- 🧠 **Dual Models**: Uses both HuggingFace and a custom-trained CNN model for accurate results. \n
- 📊 **Model Performance Dashboard**: Shows accuracy and loss metrics for each model. \n
- 📬 **Email Results**: Get your plant analysis emailed to you for easy access later. \n
- 🔍 **Searchable Diagnosis Page**: Quickly search through plant disease records and results. \n
- 🗂 **Grouped History**: View all your plant analyses grouped by individual plants. \n
- 🌱 **Plant Care Tips**: Get tips on how to treat or manage specific diseases. \n
- 🤖 **AI Support Chat**: Ask questions like this anytime — LeafLogic Assistant is here to help! \n
""".strip()
    
    # Static step-by-step guide for uploading an image
    hardcoded_upload_reply = """
To upload an image of your plant for disease detection, follow these steps:
1. Go to the **Diagnosis Page** on your LeafLogic app.
2. Look for the **'Upload Image'** button and click it.
3. Select an image of your plant that you'd like to analyze.
4. Wait for the system to process the image and display the disease analysis results.
5. Optionally, you can email the results to yourself by entering your email after the analysis.
"""
    
    # If the prompt asks about features
    if any(keyword in prompt for keyword in feature_keywords):
        return jsonify({"reply": hardcoded_feature_reply})
    
    # If the prompt asks about uploading an image
    elif any(keyword in prompt for keyword in upload_keywords):
        return jsonify({"reply": hardcoded_upload_reply})

    # Using deepseek for all other questions
    try:
        response = ollama.chat(
            model="deepseek-r1:1.5b",  
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the AI assistant for *LeafLogic*, a web app that helps farmers identify plant diseases from images. "
                        "The app includes features like image-based disease detection, dual-model predictions, email reporting, search tools, disease history tracking, and farming support. "
                        "You are NOT related to any ERP software or cannabis industry product. "
                        "If the user asks about the app's features, respond with a bulleted list of them as clearly and helpfully as possible. "
                        "Always stay in the farming context. Remove the thinking from your response. Give short answers"
                    )
                },
                {"role": "user", "content": prompt}
            ]
        )
        
        # Remove <think> sections from the response
        cleaned_response = re.sub(r'<think>.*?</think>', '', response["message"]["content"], flags=re.DOTALL)
        
        return jsonify({"reply": cleaned_response})
    
    except Exception as e:
        app.logger.error(f"Ollama support error: {e}")
        return jsonify({"reply": "Sorry, something went wrong while trying to help."}), 500

def save_review_to_file(review):
    try:
        if not os.path.exists(REVIEW_FILE):
            with open(REVIEW_FILE, "w") as f:
                json.dump([], f)  # Initialize an empty list if the file doesn't exist

        # Load existing reviews and add the new review
        with open(REVIEW_FILE, "r") as f:
            reviews = json.load(f)

        reviews.append(review)

        # Save the updated list of reviews back to the file
        with open(REVIEW_FILE, "w") as f:
            json.dump(reviews, f, indent=4)

        return True
    except Exception as e:
        app.logger.error(f"Failed to save review: {str(e)}")
        return False


        return True
    except Exception as e:
        app.logger.error(f"Failed to save review: {str(e)}")
        return False

@app.route('/submit-review', methods=['POST'])
def submit_review():
    try:
        name = request.form.get('name', 'Anonymous')
        message = request.form.get('message', '')
        image_file = request.files.get('profileImage')
        photo_url = request.form.get('photo')  # <-- this was missing before

        if not message:
            return jsonify({"success": False, "error": "Review message is required"}), 400

        # Determine which image to use: uploaded file or avatar URL
        if image_file and allowed_file(image_file.filename):
            filename = secure_filename(f"{int(time.time())}_{image_file.filename}")
            filepath = os.path.join('static/reviews', filename)
            try:
                image_file.save(filepath)
                image_url = f"/static/reviews/{filename}"
            except Exception as e:
                app.logger.error(f"Failed to save image: {str(e)}")
                image_url = "/static/reviews/default_user.png"
        else:
            image_url = photo_url or "/static/reviews/default_user.png"

        review_data = {
            "name": name,
            "message": message,
            "image": image_url,
            "timestamp": time.time()
        }

        if save_review_to_file(review_data):
            return jsonify({"success": True, "review": review_data}), 200
        else:
            return jsonify({"success": False, "error": "Failed to save review"}), 500

    except Exception as e:
        app.logger.error(f"Review submission error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/get-reviews', methods=['GET'])
def get_reviews():
    try:
        # Load the reviews from the file every time the endpoint is hit
        with open(REVIEW_FILE, "r") as f:
            reviews = json.load(f)
        return jsonify(reviews)
    except Exception as e:
        app.logger.error(f"Failed to load reviews: {str(e)}")
        return jsonify([]), 500

# Admin login decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth = request.authorization
        if not auth or not (auth.username == ADMIN_CREDENTIALS["username"] and auth.password == ADMIN_CREDENTIALS["password"]):
            return jsonify({"success": False, "error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/admin/login', methods=['POST'])
def admin_login():
    data = request.get_json()
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({"success": False, "error": "Username and password required"}), 400
    
    if data['username'] == ADMIN_CREDENTIALS["username"] and data['password'] == ADMIN_CREDENTIALS["password"]:
        return jsonify({"success": True, "message": "Login successful"})
    else:
        return jsonify({"success": False, "error": "Invalid credentials"}), 401

@app.route('/testimonials', methods=['DELETE'])
@admin_required
def delete_all_reviews():
    try:
        with open(REVIEW_FILE, "w") as f:
            json.dump([], f)
        app.logger.info("All reviews successfully deleted.")

        # Confirm deletion
        with open(REVIEW_FILE, "r") as f:
            reviews = json.load(f)
            app.logger.info(f"File contents after deletion: {reviews}")

        return jsonify({"success": True, "message": "All reviews deleted"})
    except Exception as e:
        app.logger.error(f"Failed to delete reviews: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    try:
        classifier.load_model()
        app.run(host="0.0.0.0", port=5000, debug=False)
    except Exception as e:
        app.logger.critical(f"Failed to start: {str(e)}")