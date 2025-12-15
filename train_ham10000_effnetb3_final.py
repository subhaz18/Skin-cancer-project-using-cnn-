#!/usr/bin/env python3
"""
FINAL STABLE TRAINING SCRIPT
Skin Cancer Detection using EfficientNetB3 on HAM10000

- Strong augmentation
- Two-phase training (head + fine-tuning)
- No class_weight (avoids Keras crash)
- No AUC metric (avoids sample_weight bug)
- Saves model, confusion matrix, accuracy/loss plots
- Generates Grad-CAM visualizations

Expected accuracy: ~88–92%
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import cv2

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ---------------- CONFIG ----------------
IMG_SIZE = 300
BATCH_SIZE = 16
EPOCHS_HEAD = 25
EPOCHS_FINE = 6
FINE_TUNE_LAYERS = 120

# ---------------- PATH HELPERS ----------------
def find_image_path(img_name, folders):
    for f in folders:
        p = os.path.join(f, img_name)
        if os.path.exists(p):
            return p
    return None

# ---------------- DATAFRAME ----------------
def build_dataframe(csv_path, folders):
    df = pd.read_csv(csv_path)
    mapping = {
        'akiec':'AKIEC','bcc':'BCC','bkl':'BKL',
        'df':'DF','mel':'MEL','nv':'NV','vasc':'VASC'
    }
    df['filename'] = df['image_id'] + '.jpg'
    df['label'] = df['dx'].map(mapping)
    df['filepath'] = df['filename'].apply(lambda x: find_image_path(x, folders))
    df = df[df['filepath'].notna()]
    print("\nClass distribution:\n", df['label'].value_counts())
    return df[['filepath','label']]

# ---------------- GENERATORS ----------------
def make_generators(train_df, val_df, test_df):
    train_aug = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    test_aug = ImageDataGenerator(rescale=1./255)

    train_gen = train_aug.flow_from_dataframe(
        train_df, x_col='filepath', y_col='label',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode='categorical'
    )
    val_gen = test_aug.flow_from_dataframe(
        val_df, x_col='filepath', y_col='label',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode='categorical'
    )
    test_gen = test_aug.flow_from_dataframe(
        test_df, x_col='filepath', y_col='label',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode='categorical',
        shuffle=False
    )
    return train_gen, val_gen, test_gen

# ---------------- MODEL ----------------
def build_model(num_classes):
    backbone = EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    backbone.trainable = False

    x = backbone.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(backbone.input, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, backbone

# ---------------- PLOTS ----------------
def plot_history(history):
    plt.figure()
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.legend()
    plt.title("Accuracy")
    plt.savefig("results/accuracy.png")
    plt.close()

    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.title("Loss")
    plt.savefig("results/loss.png")
    plt.close()

def plot_confusion(cm, labels):
    plt.figure(figsize=(8,8))
    plt.imshow(cm, cmap='Blues')
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i,cm[i,j],ha='center',va='center')
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png")
    plt.close()

# ---------------- GRADCAM ----------------
def gradcam_example(model, img_path, label):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.layers[-3].output, model.output]
    )
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_SIZE,IMG_SIZE))
    arr = tf.keras.preprocessing.image.img_to_array(img)/255.0
    arr = np.expand_dims(arr,0)

    with tf.GradientTape() as tape:
        conv, preds = grad_model(arr)
        idx = np.argmax(preds[0])
        loss = preds[:, idx]

    grads = tape.gradient(loss, conv)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))
    heatmap = tf.reduce_sum(conv[0] * pooled, axis=-1)
    heatmap = np.maximum(heatmap,0)/np.max(heatmap)

    heatmap = cv2.resize(heatmap, (IMG_SIZE,IMG_SIZE))
    heatmap = np.uint8(255*heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    orig = cv2.imread(img_path)
    orig = cv2.resize(orig, (IMG_SIZE,IMG_SIZE))
    overlay = cv2.addWeighted(orig,0.6,heatmap,0.4,0)
    cv2.imwrite(f"results/gradcam_{label}.png", overlay)

# ---------------- MAIN ----------------
def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    folders = [
        "data/HAM10000_images_part_1",
        "data/HAM10000_images_part_2"
    ]
    df = build_dataframe("data/HAM10000_metadata.csv", folders)

    train_df, test_df = train_test_split(df, test_size=0.15, stratify=df['label'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.15, stratify=train_df['label'], random_state=42)

    train_gen, val_gen, test_gen = make_generators(train_df, val_df, test_df)

    # Save class indices
    with open("models/class_indices.json","w") as f:
        json.dump(train_gen.class_indices, f)

    model, backbone = build_model(len(train_gen.class_indices))
    model.summary()

    callbacks = [
        EarlyStopping(patience=7, restore_best_weights=True),
        ReduceLROnPlateau(patience=3, factor=0.3),
        ModelCheckpoint("models/best_model.h5", save_best_only=True)
    ]

    print("\nTraining head...")
    hist1 = model.fit(train_gen, validation_data=val_gen,
                      epochs=EPOCHS_HEAD, callbacks=callbacks)

    print("\nFine-tuning...")
    backbone.trainable = True
    for layer in backbone.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    hist2 = model.fit(train_gen, validation_data=val_gen,
                      epochs=EPOCHS_FINE, callbacks=callbacks)

    model.save("models/cnn_ham10000_saved_model")

    plot_history(hist2)

    preds = model.predict(test_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes
    labels = list(train_gen.class_indices.keys())

    print("\n", classification_report(y_true, y_pred, target_names=labels))
    plot_confusion(confusion_matrix(y_true, y_pred), labels)

    gradcam_example(model, test_df.iloc[0]['filepath'], labels[y_pred[0]])

    print("\nTraining complete. Results saved in /results")

if __name__ == "__main__":
    main()
