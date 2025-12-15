#!/usr/bin/env python3
"""
Flask app that:
- loads saved model (models/cnn_ham10000_saved_model by default)
- loads models/class_indices.json to map indices -> labels
- serves index.html for upload and result.html to show predictions
- optionally generates Grad-CAM overlay images and serves them
- logs feedback (user_label + doctor_confirmed) into data/user_feedback.csv
"""

import os
import uuid
import csv
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2

# ---------- CONFIG ----------
MODEL_PATH = "models/cnn_ham10000_saved_model"    # change to updated model path if needed
CLASS_INDICES_PATH = "models/class_indices.json"
UPLOAD_FOLDER = "static/uploads"
FEEDBACK_CSV = "data/user_feedback.csv"
IMG_SIZE = (300, 300)   # must match training image size (300 for EfficientNetB3)
MODE = "multiclass"     # 'multiclass' or 'binary'
ALLOW_GRADCAM = True    # set False to disable Grad-CAM option

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(FEEDBACK_CSV), exist_ok=True)

# ---------- Load model ----------
print("Loading model from:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

# ---------- Load class indices mapping ----------
if os.path.exists(CLASS_INDICES_PATH):
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)   # label -> idx
    # build idx -> label list ordered by index
    inv = {int(v): k for k, v in class_indices.items()}
    idx2label = [inv[i] for i in range(len(inv))]
    print("Loaded class indices:", class_indices)
else:
    # fallback: you must set labels appropriately
    print("Warning: class_indices.json missing. Using fallback labels.")
    idx2label = ["AKIEC","BCC","BKL","DF","MEL","NV","VASC"]

# ---------- Utils ----------
def preprocess_image(path):
    img = load_img(path, target_size=IMG_SIZE)
    arr = img_to_array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(path):
    x = preprocess_image(path)
    preds = model.predict(x)[0]
    if MODE == "binary":
        prob = float(preds[0])
        label = "MEL" if prob >= 0.5 else "OTHER"
        conf = prob if prob >= 0.5 else 1 - prob
        return label, conf, preds
    else:
        idx = int(np.argmax(preds))
        label = idx2label[idx]
        conf = float(preds[idx])
        return label, conf, preds

# ---------- Grad-CAM helpers ----------
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) or 'conv' in layer.name.lower():
            return layer.name
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def save_gradcam_overlay(img_path, out_path, heatmap, alpha=0.4, target_size=IMG_SIZE):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError("Image not found for gradcam: " + img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size[0], target_size[1]))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = heatmap_color * alpha + img
    cv2.imwrite(out_path, cv2.cvtColor(superimposed.astype(np.uint8), cv2.COLOR_RGB2BGR))

# ---------- Feedback logging ----------
def init_feedback_csv():
    if not os.path.exists(FEEDBACK_CSV):
        with open(FEEDBACK_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp","image_filename","model_mode","model_pred_label","model_confidence","user_label","doctor_confirmed"])

def log_feedback(image_filename, model_label, model_conf, user_label, doctor_confirmed):
    init_feedback_csv()
    with open(FEEDBACK_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), image_filename, MODE, model_label, f"{model_conf:.4f}", user_label or "", "True" if doctor_confirmed else "False"])

# ---------- Flask app ----------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8 MB max upload

@app.route('/', methods=['GET','POST'])
def index():
    possible_labels = idx2label if MODE=='multiclass' else ["MEL","OTHER"]
    if request.method == 'POST':
        if 'image' not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash("No selected file")
            return redirect(request.url)
        ext = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        pred_label, conf, preds = predict_image(save_path)

        # Grad-CAM if requested and supported
        gradcam_file = None
        if ALLOW_GRADCAM and request.form.get('gradcam') == 'on' and MODE == 'multiclass':
            try:
                last_conv = find_last_conv_layer(model)
                if last_conv is not None:
                    img_arr = preprocess_image(save_path)
                    pred_idx = int(np.argmax(preds))
                    heatmap = make_gradcam_heatmap(img_arr, model, last_conv, pred_index=pred_idx)
                    gradcam_fname = f"gradcam_{uuid.uuid4().hex}.png"
                    gradcam_path = os.path.join(UPLOAD_FOLDER, gradcam_fname)
                    save_gradcam_overlay(save_path, gradcam_path, heatmap, alpha=0.4, target_size=IMG_SIZE)
                    gradcam_file = gradcam_fname
                else:
                    print("Grad-CAM: no conv layer found")
            except Exception as e:
                print("Grad-CAM generation failed:", e)

        user_label = request.form.get('user_label','').strip()
        doctor_confirmed = (request.form.get('doctor_confirmed') == 'on')
        log_feedback(filename, pred_label, conf, user_label, doctor_confirmed)

        return render_template('result.html',
                               image_file=filename,
                               pred_label=pred_label,
                               confidence=f"{conf*100:.2f}",
                               user_label=user_label,
                               doctor_confirmed=doctor_confirmed,
                               gradcam_file=gradcam_file)

    return render_template('index.html', possible_labels=possible_labels, mode=MODE)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
