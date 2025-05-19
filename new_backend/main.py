import os
import io
import time
import torch
import logging
from PIL import Image, ImageDraw
from flask import Flask, request, jsonify
from flask_cors import CORS
from logging.handlers import RotatingFileHandler
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchvision import transforms
from werkzeug.utils import secure_filename
import pillow_heif
import ollama
import re
import json
from functools import wraps
from dotenv import load_dotenv
from new_backend.email import send_email
from werkzeug.middleware.proxy_fix import ProxyFix

# Initialize environment
load_dotenv()
REVIEW_FILE = "new_backend/reviews.json"
pillow_heif.register_heif_opener()

# Flask app setup
app = Flask(__name__)
CORS(app)

# Configuration
app.config.update(
    {
        "MAX_CONTENT_LENGTH": 10 * 1024 * 1024,
        "ALLOWED_EXTENSIONS": {"png", "jpg", "jpeg", "gif", "heic", "heif"},
        "MODEL_CACHE": "./model_cache",
        "THROTTLE_LIMIT": 5,
    }
)

# Model configuration
MODEL_CONFIG = {
    "general_model": "google/vit-base-patch16-224",
    "specialized_model": "DunnBC22/vit-base-Plantsv1-Disease-Classification",  # Verified plant disease model
    "fallback_model": "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
}

# Admin credentials
ADMIN_CREDENTIALS = {
    "username": os.getenv("ADMIN_USERNAME", "admin"),
    "password": os.getenv("ADMIN_PASSWORD", "leafadmin123"),
}

# Setup directories
os.makedirs("static/reviews", exist_ok=True)
if not os.path.exists("static/reviews/default_user.png"):
    img = Image.new("RGB", (200, 200), color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    draw.text((50, 80), "User", fill=(0, 0, 0))
    img.save("static/reviews/default_user.png")

# Logging configuration
handler = RotatingFileHandler("app.log", maxBytes=10000, backupCount=3)
handler.setFormatter(
    logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]"
    )
)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)


class PlantDiseaseClassifier:
    def __init__(self):
        self.models = {}
        self.processors = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_config = {
            "general": "google/vit-base-patch16-224",
            "specialized": "DunnBC22/vit-base-Plantsv1-Disease-Classification",
            "fallback": "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
        }
        app.logger.info(f"Using device: {self.device}")

    def load_model(self):
        if self.models:
            return

        try:
            start_time = time.time()
            app.logger.info("Loading models...")

            # Try loading specialized model first
            try:
                self.processors["specialized"] = AutoImageProcessor.from_pretrained(
                    self.model_config["specialized"],
                    cache_dir=app.config["MODEL_CACHE"],
                )
                self.models["specialized"] = (
                    AutoModelForImageClassification.from_pretrained(
                        self.model_config["specialized"],
                        cache_dir=app.config["MODEL_CACHE"],
                    )
                    .to(self.device)
                    .eval()
                )
                app.logger.info("Specialized plant disease model loaded")
            except Exception as e:
                app.logger.warning(f"Failed to load specialized model: {str(e)}")
                self.models["specialized"] = None

            # Load general vision model
            self.processors["general"] = AutoImageProcessor.from_pretrained(
                self.model_config["general"], cache_dir=app.config["MODEL_CACHE"]
            )
            self.models["general"] = (
                AutoModelForImageClassification.from_pretrained(
                    self.model_config["general"], cache_dir=app.config["MODEL_CACHE"]
                )
                .to(self.device)
                .eval()
            )
            app.logger.info("General vision model loaded")

            # Load fallback model
            try:
                self.processors["fallback"] = AutoImageProcessor.from_pretrained(
                    self.model_config["fallback"], cache_dir=app.config["MODEL_CACHE"]
                )
                self.models["fallback"] = (
                    AutoModelForImageClassification.from_pretrained(
                        self.model_config["fallback"],
                        cache_dir=app.config["MODEL_CACHE"],
                    )
                    .to(self.device)
                    .eval()
                )
                app.logger.info("Fallback plant disease model loaded")
            except Exception as e:
                app.logger.warning(f"Failed to load fallback model: {str(e)}")
                self.models["fallback"] = None

            if not any(model for model in self.models.values()):
                raise RuntimeError("All model loading attempts failed")

            app.logger.info(f"Models loaded in {time.time()-start_time:.2f}s")

        except Exception as e:
            app.logger.error(f"Model loading failed: {str(e)}")
            raise RuntimeError(f"Could not load any models: {str(e)}")

    def preprocess_image(self, image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")
            return image
        except Exception as e:
            app.logger.error(f"Image processing failed: {str(e)}")
            raise ValueError("Invalid or unsupported image file")

    def _predict_with_model(self, image, model_key):
        try:
            processor = self.processors[model_key]
            model = self.models[model_key]

            inputs = processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                pred_idx = outputs.logits.argmax(-1).item()
                confidence = float(probs[0][pred_idx].item())
                label = model.config.id2label[pred_idx]

            return {
                "prediction": label,
                "confidence": confidence,
                "model_key": model_key,
            }
        except Exception as e:
            app.logger.error(f"Prediction with {model_key} model failed: {str(e)}")
            raise

    def predict(self, image_bytes, plant_type=None):
        try:
            self.load_model()
            image = self.preprocess_image(image_bytes)

            plant_type = plant_type.strip().lower() if plant_type else ""
            use_specialized = plant_type in ["pepper bell", "tomato", "potato"]

            # Try specialized model first if applicable
            if use_specialized and self.models.get("specialized"):
                try:
                    result = self._predict_with_model(image, "specialized")
                    result["model_used"] = "Specialized Plant Model"
                    return {
                        "success": True,
                        **result,
                        "confidence": int(result["confidence"] * 100),
                    }
                except Exception:
                    app.logger.warning("Falling back to general model")

            # Try general model
            if self.models.get("general"):
                try:
                    result = self._predict_with_model(image, "general")
                    result["model_used"] = "General Vision Model"
                    return {
                        "success": True,
                        **result,
                        "confidence": int(result["confidence"] * 100),
                    }
                except Exception:
                    app.logger.warning("Falling back to mobile model")

            # Final fallback
            if self.models.get("fallback"):
                result = self._predict_with_model(image, "fallback")
                result["model_used"] = "Fallback Mobile Model"
                return {
                    "success": True,
                    **result,
                    "confidence": int(result["confidence"] * 100),
                }

            raise RuntimeError("All prediction attempts failed")

        except Exception as e:
            app.logger.error(f"Prediction error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "Could not process image",
            }

    def preprocess_image(self, image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes))
            return image.convert("RGB") if image.mode != "RGB" else image
        except Exception as e:
            app.logger.error(f"Image processing failed: {str(e)}")
            raise ValueError("Invalid or unsupported image file")

    def predict(self, image_bytes, plant_type):
        try:
            self.load_model()
            image = self.preprocess_image(image_bytes)

            plant_type = plant_type.strip().lower()

            # Determine which model to use
            use_specialized = plant_type in ["pepper bell", "tomato", "potato"]
            model_key = "specialized" if use_specialized else "general"

            processor = self.processors[model_key]
            model = self.models[model_key]

            inputs = processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                pred_idx = outputs.logits.argmax(-1).item()
                confidence = float(probs[0][pred_idx].item())
                label = model.config.id2label[pred_idx]

            # Post-process label for consistency
            if use_specialized and "___" in label:
                label = label.replace("___", "__")  # Standardize separator

            return {
                "success": True,
                "model_used": "Specialized" if use_specialized else "General",
                "prediction": label,
                "confidence": int(confidence * 100),
            }

        except Exception as e:
            # Fallback to simpler model if available
            try:
                if "fallback" not in self.models:
                    self.processors["fallback"] = AutoImageProcessor.from_pretrained(
                        MODEL_CONFIG["fallback_model"],
                        cache_dir=app.config["MODEL_CACHE"],
                    )
                    self.models["fallback"] = (
                        AutoModelForImageClassification.from_pretrained(
                            MODEL_CONFIG["fallback_model"],
                            cache_dir=app.config["MODEL_CACHE"],
                        )
                        .to(self.device)
                        .eval()
                    )

                inputs = self.processors["fallback"](
                    images=image, return_tensors="pt"
                ).to(self.device)
                with torch.no_grad():
                    outputs = self.models["fallback"](**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    pred_idx = outputs.logits.argmax(-1).item()
                    confidence = float(probs[0][pred_idx].item())
                    label = self.models["fallback"].config.id2label[pred_idx]

                return {
                    "success": True,
                    "model_used": "Fallback",
                    "prediction": label,
                    "confidence": int(confidence * 100),
                }
            except Exception as fallback_error:
                app.logger.error(
                    f"Prediction error: {str(e)} | Fallback failed: {str(fallback_error)}"
                )
                return {"success": False, "error": f"Prediction failed: {str(e)}"}


# Initialize classifier
classifier = PlantDiseaseClassifier()


# Helper functions
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


def ensure_reviews_file():
    if not os.path.exists(REVIEW_FILE):
        with open(REVIEW_FILE, "w") as f:
            json.dump([], f)
    else:
        try:
            with open(REVIEW_FILE, "r") as f:
                json.load(f)
        except json.JSONDecodeError:
            with open(REVIEW_FILE, "w") as f:
                json.dump([], f)


ensure_reviews_file()


# Routes
@app.route("/health", methods=["GET"])
def health_check():
    try:
        classifier.load_model()
        return (
            jsonify(
                {
                    "status": "healthy",
                    "models_loaded": bool(
                        classifier.models
                    ),  # Check if any models are loaded
                }
            ),
            200,
        )
    except Exception as e:
        return (
            jsonify({"status": "unhealthy", "error": str(e), "models_loaded": False}),
            500,
        )


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files or "plantType" not in request.form:
        return jsonify({"success": False, "error": "Missing image or plantType"}), 400

    file = request.files["image"]
    plant_type = request.form["plantType"]

    if file and allowed_file(file.filename):
        try:
            image_bytes = file.read()
            return jsonify(classifier.predict(image_bytes, plant_type))
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
    return jsonify({"success": False, "error": "Invalid file type"}), 400


@app.route("/accuracy/chart", methods=["GET"])
def get_chart_data():
    return jsonify(
        {
            "success": True,
            "data": {"accuracies": history["accuracies"], "losses": history["losses"]},
        }
    )


@app.route("/send-results", methods=["POST"])
def send_results():
    data = request.get_json()
    email = data.get("email")
    results = data.get("results", [])
    good_results = [
        {"name": r["name"], "confidence": r["confidence"], "prediction": r["status"]}
        for r in results
    ]

    message = data.get(
        "message",
        f"""<html><body><h1>Your LeafLogic report is ready!</h1><table><tr><th>Name</th><th>Confidence</th><th>Status</th></tr>{''.join(f"<tr><td>{res['name']}</td><td>{res['confidence']}</td><td>{res['prediction']}</td></tr>" for res in good_results)}</table><p>Thank you for using LeafLogic!</p><p>Best regards,<br>LeafLogic Team</p></body></html>""",
    )

    if not email:
        return jsonify({"success": False, "error": "Email is required"}), 400

    success = send_email(email, message)
    return jsonify(
        {"success": success, "error": None if success else "Failed to send email"}
    ), (500 if not success else 200)


@app.route("/ollama-support", methods=["POST"])
def ollama_support():
    data = request.get_json()
    prompt = data.get("prompt", "").strip().lower()

    # Static responses for certain steps like uploading images or feature-related questions
    feature_keywords = [
        "features",
        "what can you do",
        "capabilities",
        "app features",
        "how can you help",
        "what's good",
        "functions",
        "tools",
        "what is leaflogic",
        "tell me about leaflogic",
        "leaflogic features",
    ]

    upload_keywords = ["upload", "how to upload", "upload an image", "how do I upload"]

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
            model="qwen2.5:0.5b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the AI assistant for *LeafLogic*, a web app that helps farmers identify plant diseases from images. "
                        "The app includes features like image-based disease detection, dual-model predictions, email reporting, search tools, disease history tracking, and farming support. "
                        "You are NOT related to any ERP software or cannabis industry product. "
                        "If the user asks about the app's features, respond with a bulleted list of them as clearly and helpfully as possible. "
                        "Always stay in the farming context. Remove the thinking from your response. Give short answers"
                        "If the user asks about how to help in terms of fixing a problem with their plant, then give them an answer on how to fix it"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )

        # Remove <think> sections from the response
        cleaned_response = re.sub(
            r"<think>.*?</think>", "", response["message"]["content"], flags=re.DOTALL
        )

        return jsonify({"reply": cleaned_response})

    except Exception as e:
        app.logger.error(f"Ollama support error: {e}")
        return (
            jsonify({"reply": "Sorry, something went wrong while trying to help."}),
            500,
        )


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


@app.route("/submit-review", methods=["POST"])
def submit_review():
    try:
        name = request.form.get("name", "Anonymous")
        message = request.form.get("message", "")
        image_file = request.files.get("profileImage")
        photo_url = request.form.get("photo")  # <-- this was missing before

        if not message:
            return (
                jsonify({"success": False, "error": "Review message is required"}),
                400,
            )

        # Determine which image to use: uploaded file or avatar URL
        if image_file and allowed_file(image_file.filename):
            filename = secure_filename(f"{int(time.time())}_{image_file.filename}")
            filepath = os.path.join("static/reviews", filename)
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
            "timestamp": time.time(),
        }

        if save_review_to_file(review_data):
            return jsonify({"success": True, "review": review_data}), 200
        else:
            return jsonify({"success": False, "error": "Failed to save review"}), 500

    except Exception as e:
        app.logger.error(f"Review submission error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/get-reviews", methods=["GET"])
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
        if not auth or not (
            auth.username == ADMIN_CREDENTIALS["username"]
            and auth.password == ADMIN_CREDENTIALS["password"]
        ):
            return jsonify({"success": False, "error": "Authentication required"}), 401
        return f(*args, **kwargs)

    return decorated_function


@app.route("/admin/login", methods=["POST"])
def admin_login():
    data = request.get_json()
    if not data or "username" not in data or "password" not in data:
        return (
            jsonify({"success": False, "error": "Username and password required"}),
            400,
        )

    if (
        data["username"] == ADMIN_CREDENTIALS["username"]
        and data["password"] == ADMIN_CREDENTIALS["password"]
    ):
        return jsonify({"success": True, "message": "Login successful"})
    else:
        return jsonify({"success": False, "error": "Invalid credentials"}), 401


@app.route("/testimonials", methods=["DELETE"])
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
