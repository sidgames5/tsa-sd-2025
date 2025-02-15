from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from backend.model import analyze_image  # Import AI function

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/upload", methods=["POST"])
def upload_file():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Invalid file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Analyze image with AI
    result = analyze_image(filepath)

    return jsonify({"message": f"Detected: {result}"})


if __name__ == "__main__":
    app.run(debug=True)
