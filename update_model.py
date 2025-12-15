#!/usr/bin/env python3
"""
Offline fine-tune script that:
- reads data/user_feedback.csv
- filters rows with doctor_confirmed == True and non-empty user_label
- builds a small ImageDataGenerator from static/uploads/
- fine-tunes the saved model (models/cnn_ham10000_saved_model)
- saves updated model to models/cnn_ham10000_updated_model
"""

import os
import json
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# -------------- CONFIG --------------
FEEDBACK_CSV = "data/user_feedback.csv"
UPLOAD_FOLDER = "static/uploads"
ORIG_MODEL = "models/cnn_ham10000_saved_model"
UPDATED_MODEL = "models/cnn_ham10000_updated_model"
CLASS_INDICES_PATH = "models/class_indices.json"
IMG_SIZE = (300, 300)
BATCH_SIZE = 8
EPOCHS = 5     # fine-tune epochs (small)
LR = 1e-5      # very low LR for safe fine-tune
MODE = "multiclass"  # 'multiclass' or 'binary'

# -------------- Helpers --------------
def load_feedback():
    if not os.path.exists(FEEDBACK_CSV):
        print("No feedback CSV found:", FEEDBACK_CSV)
        return pd.DataFrame()
    df = pd.read_csv(FEEDBACK_CSV)
    # normalize doctor_confirmed to boolean
    df['doctor_confirmed'] = df['doctor_confirmed'].astype(str).str.lower().isin(['true','1','yes','y'])
    df['user_label'] = df['user_label'].astype(str).str.strip()
    df = df[(df['doctor_confirmed']) & (df['user_label'] != '')].copy()
    return df

def build_feedback_df(df, allowed_labels):
    # Keep entries with allowed_labels and ensure file exists
    df = df[df['user_label'].isin(allowed_labels)].copy()
    df['filepath'] = df['image_filename'].apply(lambda x: os.path.join(UPLOAD_FOLDER, x))
    df = df[df['filepath'].apply(os.path.exists)].copy()
    return df[['filepath','user_label']].rename(columns={'user_label':'label'})

def create_generator(feedback_df):
    datagen = ImageDataGenerator(rescale=1./255.,
                                 rotation_range=10,
                                 zoom_range=0.08,
                                 horizontal_flip=True,
                                 fill_mode='nearest')
    gen = datagen.flow_from_dataframe(feedback_df,
                                     x_col='filepath',
                                     y_col='label',
                                     target_size=IMG_SIZE,
                                     batch_size=BATCH_SIZE,
                                     class_mode='categorical' if MODE=='multiclass' else 'binary',
                                     shuffle=True)
    return gen

# -------------- Main --------------
def main():
    print("Loading feedback CSV...")
    df = load_feedback()
    if df.empty:
        print("No confirmed feedback samples found. Exiting.")
        return

    # Load class indices to know allowed labels (mapping label->idx)
    if not os.path.exists(CLASS_INDICES_PATH):
        print("class_indices.json not found:", CLASS_INDICES_PATH)
        print("Cannot proceed without label mapping from training. Exiting.")
        return
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    allowed_labels = list(class_indices.keys())
    print("Allowed labels:", allowed_labels)

    fb_df = build_feedback_df(df, allowed_labels)
    if fb_df.empty:
        print("No feedback images found with allowed labels. Exiting.")
        return

    print("Feedback samples count:", len(fb_df))
    gen = create_generator(fb_df)

    # Load original model
    print("Loading original model:", ORIG_MODEL)
    model = tf.keras.models.load_model(ORIG_MODEL)
    print("Model loaded.")

    # Compile with low LR
    loss = 'categorical_crossentropy' if MODE=='multiclass' else 'binary_crossentropy'
    model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss=loss, metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2),
        ModelCheckpoint('models/ft_temp.h5', monitor='loss', save_best_only=True)
    ]

    steps = max(1, gen.samples // BATCH_SIZE)
    print(f"Starting fine-tune for {EPOCHS} epochs ({steps} steps per epoch)...")
    model.fit(gen, epochs=EPOCHS, steps_per_epoch=steps, callbacks=callbacks)

    # Save updated model
    model.save(UPDATED_MODEL)
    print("Updated model saved to:", UPDATED_MODEL)
    print("Done.")

if __name__ == '__main__':
    main()
