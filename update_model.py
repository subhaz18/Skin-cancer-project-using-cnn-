import os
import json
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------- CONFIG ----------------
MODEL_PATH = "models/cnn_ham10000_saved_model.keras"
UPDATED_MODEL_PATH = "models/cnn_ham10000_updated_model.keras"
FEEDBACK_CSV = "data/user_feedback.csv"
IMG_SIZE = 300

if not os.path.exists(FEEDBACK_CSV):
    print("No feedback file found. Exiting.")
    exit()

df = pd.read_csv(FEEDBACK_CSV)

# Only verified feedback
df = df[(df["doctor_confirmed"] == True) & (df["user_label"].notna())]

if len(df) < 10:
    print("Not enough verified samples to update model.")
    exit()

datagen = ImageDataGenerator(rescale=1./255)

gen = datagen.flow_from_dataframe(
    df,
    x_col="image_path",
    y_col="user_label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=4,
    class_mode="categorical"
)

model = tf.keras.models.load_model(MODEL_PATH)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(gen, epochs=3)
model.save(UPDATED_MODEL_PATH)

print("Updated model saved successfully.")
