import tensorflow as tf
import numpy as np
import cv2
import json
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ================= CONFIG =================
IMG_SIZE = 300
CLASS_INDEX_PATH = "models/class_indices.json"

# Load class indices (label → index)
with open(CLASS_INDEX_PATH, "r") as f:
    class_indices = json.load(f)

# Reverse mapping (index → label)
idx2label = {v: k for k, v in class_indices.items()}

# ================= UTILITIES =================
def get_last_conv_layer(model):
    """
    Returns the name of the last Conv2D layer.
    Required for correct Grad-CAM.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model.")

# ================= GRAD-CAM CORE =================
def generate_gradcam(model, image_path, output_path):
    """
    Generates Grad-CAM visualization and saves it.

    Args:
        model        : Trained Keras model
        image_path  : Path to original image
        output_path : Path to save Grad-CAM overlay
    """

    # Identify last conv layer
    last_conv_layer_name = get_last_conv_layer(model)

    # Load & preprocess image
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Create gradient model
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling on gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Compute heatmap
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    heatmap = heatmap.numpy()

    # Load original image for overlay
    original_img = cv2.imread(image_path)
    original_img = cv2.resize(original_img, (IMG_SIZE, IMG_SIZE))

    # Resize & colorize heatmap
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on image
    overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    # Save result
    cv2.imwrite(output_path, overlay)

    # Return predicted label (optional)
    predicted_label = idx2label[int(class_idx)]
    return predicted_label
