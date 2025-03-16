from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import torch
from torchvision import transforms
from PIL import Image
from backend.model import PlantDiseaseModel
from backend.main_analyze import train_dataset

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Constants
IMG_SIZE = (224, 224)
CLASS_NAMES = list(train_dataset.class_to_idx.keys())
print(f"Updated CLASS_NAMES: {CLASS_NAMES}")


# Load trained model
model = PlantDiseaseModel(num_classes=len(CLASS_NAMES))
model.load_state_dict(
    torch.load("models/plant_disease_model.pth", map_location=torch.device("cpu"))
)
model.eval()

# Image transformations
transform = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def analyze_image(image_path):
    # Analyze an image using the trained model and return the predicted class.
    print(f"🔍 Analyzing image: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
        print(f"Image opened successfully.")

        image = transform(image).unsqueeze(0)
        print(f"Image transformed.")

        with torch.no_grad():
            output = model(image)
            print(f"Model inference complete.")

            _, predicted = torch.max(output, 1)
            print(f"Predicted class index: {predicted.item()}")

            return CLASS_NAMES[predicted.item()]
    except Exception as e:
        print(f"Error during image analysis: {e}")
        raise e  # Re-raise error to see details in Flask logs


@flask_app.route("/api/upload", methods=["POST"])
def upload_file():
    """Flask route to handle image uploads and return predictions."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Invalid file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(flask_app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    print(f"Image saved at: {filepath}")  # Debugging print

    try:
        result = analyze_image(filepath)
        return jsonify({"message": f"Detected: {result}"})
    except Exception as e:
        print(f"Error analyzing image: {e}")  # Debugging print
        return jsonify({"error": "Failed to analyze image", "details": str(e)}), 500


image_path = "/home/sid/Code/tsa-sd/Screenshot 2025-03-15 at 8.07.23 PM.png"
result = analyze_image(image_path)
print(f"Predicted class: {result}")


if __name__ == "__main__":
    app.run(debug=True)
