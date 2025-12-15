#!/usr/bin/env python3
"""
Train script for HAM10000 (supports multiclass or binary).
Usage:
  python train_ham10000.py --data_dir data --mode multiclass --img_size 224 --batch 32 --epochs 20
"""
import json
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
# ---------- Helpers ----------
def find_image_path(img_name, folders):
    """Return full path if file exists in any folder in folders list, else None."""
    for f in folders:
        p = os.path.join(f, img_name)
        if os.path.exists(p):
            return p
    return None

def build_dataframe(metadata_csv, folders, mode):
    df = pd.read_csv(metadata_csv)
    if 'image_id' not in df.columns or 'dx' not in df.columns:
        raise ValueError("metadata csv must contain 'image_id' and 'dx' columns.")
    df['filename'] = df['image_id'].astype(str) + '.jpg'

    # Keep only files that exist in either folder
    df['full_path'] = df['filename'].apply(lambda x: find_image_path(x, folders))
    df = df[df['full_path'].notna()].copy()

    if mode == 'binary':
        df['label'] = df['dx'].apply(lambda x: 'MEL' if str(x).lower() == 'mel' else 'OTHER')
    else:
        mapping = {'mel':'MEL','nv':'NV','bcc':'BCC','akiec':'AKIEC','bkl':'BKL','df':'DF','vasc':'VASC'}
        df['label'] = df['dx'].map(lambda x: mapping.get(str(x).lower(), str(x)))

    df = df[['filename','label']].reset_index(drop=True)
    print("Total images found:", len(df))
    print(df['label'].value_counts())
    return df

def make_splits(df, test_size=0.15, val_size=0.15, random_state=42):
    train_val, test = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=random_state)
    val_rel = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=val_rel, stratify=train_val['label'], random_state=random_state)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

def create_gens(train_df, val_df, test_df, folders, img_size=(224,224), batch=32, mode='multiclass'):
    # We will use a custom generator via flow_from_dataframe but directory param expects all images in 1 folder.
    # To handle multiple folders we will create a temporary 'filepath' column with full path (works with flow_from_dataframe)
    # But ImageDataGenerator.flow_from_dataframe accepts absolute file paths if x_col contains them.
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    # create full path column by searching in folders
    train_df['filepath'] = train_df['filename'].apply(lambda x: find_image_path(x, folders))
    val_df['filepath']   = val_df['filename'].apply(lambda x: find_image_path(x, folders))
    test_df['filepath']  = test_df['filename'].apply(lambda x: find_image_path(x, folders))

    class_mode = 'binary' if mode=='binary' else 'categorical'

    train_datagen = ImageDataGenerator(
        rescale=1./255.,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect'
    )
    val_datagen = ImageDataGenerator(rescale=1./255.)
    test_datagen = ImageDataGenerator(rescale=1./255.)

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filepath',
        y_col='label',
        target_size=img_size,
        batch_size=batch,
        class_mode=class_mode,
        shuffle=True
    )
    val_gen = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='filepath',
        y_col='label',
        target_size=img_size,
        batch_size=batch,
        class_mode=class_mode,
        shuffle=False
    )
    test_gen = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='filepath',
        y_col='label',
        target_size=img_size,
        batch_size=batch,
        class_mode=class_mode,
        shuffle=False
    )
    return train_gen, val_gen, test_gen

def build_model_with_backbone(img_size=(224,224,3), num_classes=2, base_trainable=False):
    """
    Returns (model, backbone) where backbone is the pretrained base model (EfficientNetB0)
    so you can fine-tune it later safely.
    """
    # create backbone and keep a reference
    backbone = EfficientNetB0(include_top=False, input_shape=img_size, weights='imagenet')
    backbone.trainable = base_trainable  # False for initial training

    x = backbone.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.4, name="dropout_head")(x)

    if num_classes == 2:
        out = layers.Dense(1, activation='sigmoid', name="predictions")(x)
        loss = 'binary_crossentropy'
        metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
    else:
        out = layers.Dense(num_classes, activation='softmax', name="predictions")(x)
        loss = 'categorical_crossentropy'
        metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]

    model = models.Model(inputs=backbone.input, outputs=out, name="skin_cnn_with_backbone")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss, metrics=metrics)

    return model, backbone
# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--mode', choices=['multiclass','binary'], default='multiclass')
    args = parser.parse_args()

    data_dir = args.data_dir
    folder1 = os.path.join(data_dir, 'HAM10000_images_part_1')
    folder2 = os.path.join(data_dir, 'HAM10000_images_part_2')
    metadata = os.path.join(data_dir, 'HAM10000_metadata.csv')
    os.makedirs('models', exist_ok=True)

    folders = [folder1, folder2]
    print("Building dataframe...")
    df = build_dataframe(metadata, folders, args.mode)
    train_df, val_df, test_df = make_splits(df)
    print(f"Train {len(train_df)}  Val {len(val_df)}  Test {len(test_df)}")

    train_gen, val_gen, test_gen = create_gens(train_df, val_df, test_df, folders,img_size=(args.img_size,args.img_size),batch=args.batch, mode=args.mode)

    # Save class indices mapping so web app can use correct label order
    os.makedirs('models', exist_ok=True)
    class_indices = train_gen.class_indices
    with open('models/class_indices.json', 'w') as f:
        json.dump(class_indices, f)
    print("Saved class_indices.json:", class_indices)

    if args.mode == 'binary':
        num_classes = 2
    else:
        # number of classes inferred from train_gen
        num_classes = len(class_indices)

    print("Class indices:", train_gen.class_indices)

    model, backbone = build_model_with_backbone(img_size=(args.img_size,args.img_size,3),
                                           num_classes=num_classes,
                                           base_trainable=False)
    model.summary()

# ---------- Train initial head ----------
    callbacks = [
        EarlyStopping(monitor='val_auc' if 'auc' in [m.name for m in model.metrics] else 'val_loss',
                    patience=6, restore_best_weights=True, mode='max'),
        ReduceLROnPlateau(monitor='val_auc' if 'auc' in [m.name for m in model.metrics] else 'val_loss',
                        factor=0.5, patience=3, mode='max'),
        ModelCheckpoint('models/best_model.h5', save_best_only=True,
                        monitor='val_auc' if 'auc' in [m.name for m in model.metrics] else 'val_loss',
                        mode='max')
    ]

    model.fit(train_gen, validation_data=val_gen, epochs=args.epochs, callbacks=callbacks)

# ---------- Optional: fine-tune backbone (UNSAFE to do too aggressively) ----------
# Use the backbone reference — unfreeze last N layers, keep the rest frozen:
    N = 50  # number of last layers to keep trainable; tune as needed
    print(f"Fine-tuning: unfreezing last {N} layers of the backbone")

    backbone.trainable = True
# Freeze all layers except last N
    for layer in backbone.layers[:-N]:
        layer.trainable = False
    for layer in backbone.layers[-N:]:
        layer.trainable = True

# Recompile with lower LR before fine-tuning
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss=model.loss, metrics=model.metrics)

# Fine-tune for a few epochs
    fine_tune_epochs = 3
    model.fit(train_gen, validation_data=val_gen, epochs=fine_tune_epochs, callbacks=callbacks)

# ---------- Save final model ----------
    model.save('models/cnn_ham10000_saved_model')
    print("Saved to models/cnn_ham10000_saved_model")
    # Evaluate on test
    steps = int(np.ceil(test_gen.samples / test_gen.batch_size))
    preds = model.predict(test_gen, steps=steps, verbose=1)
    if args.mode == 'binary':
        y_pred = (preds.ravel() >= 0.5).astype(int)
    else:
        y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes
    print(classification_report(y_true, y_pred, target_names=list(train_gen.class_indices.keys())))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

if __name__ == '__main__':
    main()
