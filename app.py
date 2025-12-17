import os
import uuid
import json
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from gradcam import generate_gradcam

# ---------------- CONFIG ----------------
MODEL_PATH = "models/cnn_ham10000_saved_model.keras"
CLASS_INDEX_PATH = "models/class_indices.json"
UPLOAD_FOLDER = "static/uploads"
IMG_SIZE = 300

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_INDEX_PATH, "r") as f:
    class_indices = json.load(f)
idx2label = {v: k for k, v in class_indices.items()}

# ---------------- FLASK APP ----------------
app = Flask(__name__)

def preprocess_image(path):
    img = load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file.filename == "":
            return render_template("index.html", error="No file selected")

        filename = f"{uuid.uuid4().hex}.jpg"
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(image_path)

        x = preprocess_image(image_path)
        preds = model.predict(x)[0]
        class_idx = int(np.argmax(preds))
        label = idx2label[class_idx]
        confidence = float(preds[class_idx]) * 100

        gradcam_file = None
        if request.form.get("show_gradcam"):
            gradcam_file = f"gradcam_{filename}"
            gradcam_path = os.path.join(UPLOAD_FOLDER, gradcam_file)
            generate_gradcam(model, image_path, gradcam_path)

        return render_template(
            "result.html",
            image_file=filename,
            gradcam_file=gradcam_file,
            prediction=label,
            confidence=f"{confidence:.2f}"
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
