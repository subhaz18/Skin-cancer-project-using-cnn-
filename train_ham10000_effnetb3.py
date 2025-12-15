#!/usr/bin/env python3
"""
Train EfficientNetB3 on HAM10000 with:
- strong augmentation
- class weights
- 2-phase training (head + fine-tune)
- evaluation (confusion matrix + classification report)
- plots (accuracy/loss + confusion matrix)
- Grad-CAM visualization examples

Usage:
    python train_ham10000_effnetb3.py --data_dir data --img_size 300 --batch 16 --epochs 25 --fine_tune_layers 100

Adjust args as needed.
"""
import os
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import itertools

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing import image as kimage
import cv2

# -------------------------
# Helpers for file paths
# -------------------------
def find_image_path(img_name, folders):
    for f in folders:
        p = os.path.join(f, img_name)
        if os.path.exists(p):
            return p
    return None

# -------------------------
# Build dataframe from metadata
# -------------------------
def build_dataframe(metadata_csv, folders, mode):
    df = pd.read_csv(metadata_csv)
    # Expecting 'image_id' and 'dx' columns
    if 'image_id' not in df.columns or 'dx' not in df.columns:
        raise ValueError("metadata CSV must contain columns 'image_id' and 'dx'")

    df['filename'] = df['image_id'].astype(str) + '.jpg'
    df['full_path'] = df['filename'].apply(lambda x: find_image_path(x, folders))
    df = df[df['full_path'].notna()].copy()
    if mode == 'binary':
        df['label'] = df['dx'].apply(lambda x: 'MEL' if str(x).lower() == 'mel' else 'OTHER')
    else:
        mapping = {'mel':'MEL','nv':'NV','bcc':'BCC','akiec':'AKIEC','bkl':'BKL','df':'DF','vasc':'VASC'}
        df['label'] = df['dx'].map(lambda x: mapping.get(str(x).lower(), str(x)))
    df = df[['filename','label']].reset_index(drop=True)
    print("Found images:", len(df))
    print(df['label'].value_counts())
    return df

# -------------------------
# Splits
# -------------------------
def make_splits(df, test_size=0.15, val_size=0.15, random_state=42):
    train_val, test = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=random_state)
    val_rel = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=val_rel, stratify=train_val['label'], random_state=random_state)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

# -------------------------
# Generators (supports absolute paths)
# -------------------------
def create_gens(train_df, val_df, test_df, folders, img_size=(300,300), batch=16, mode='multiclass'):
    # add filepath columns (absolute paths)
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df['filepath'] = train_df['filename'].apply(lambda x: find_image_path(x, folders))
    val_df['filepath']   = val_df['filename'].apply(lambda x: find_image_path(x, folders))
    test_df['filepath']  = test_df['filename'].apply(lambda x: find_image_path(x, folders))

    if mode == 'binary':
        class_mode = 'binary'
    else:
        class_mode = 'categorical'

    train_datagen = ImageDataGenerator(
        rescale=1./255.,
        rotation_range=40,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.05,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
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

# -------------------------
# Build model + keep backbone reference
# -------------------------
def build_model_with_backbone(input_shape=(300,300,3), num_classes=7, base_trainable=False):
    backbone = EfficientNetB3(include_top=False, input_shape=input_shape, weights='imagenet')
    backbone.trainable = base_trainable

    x = backbone.output
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dropout(0.4, name='dropout_head')(x)
    if num_classes == 2:
        out = layers.Dense(1, activation='sigmoid', name='predictions')(x)
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    else:
        out = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
        loss = 'categorical_crossentropy'
        metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]

    model = models.Model(inputs=backbone.input, outputs=out, name='efficientnetb3_skin')
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss, metrics=metrics)
    return model, backbone

# -------------------------
# Confusion matrix plotting helper
# -------------------------
def plot_confusion_matrix(cm, classes, out_path, normalize=False, title='Confusion matrix'):
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    fig = plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

# -------------------------
# Grad-CAM helpers
# -------------------------
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

def save_gradcam_overlay(img_path, out_path, heatmap, alpha=0.4, target_size=(300,300)):
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to read image for gradcam:", img_path)
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size[0], target_size[1]))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = heatmap_color * alpha + img
    cv2.imwrite(out_path, cv2.cvtColor(superimposed.astype(np.uint8), cv2.COLOR_RGB2BGR))

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--img_size', type=int, default=300)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--mode', choices=['multiclass','binary'], default='multiclass')
    parser.add_argument('--fine_tune_layers', type=int, default=100)
    args = parser.parse_args()

    data_dir = args.data_dir
    folder1 = os.path.join(data_dir, 'HAM10000_images_part_1')
    folder2 = os.path.join(data_dir, 'HAM10000_images_part_2')
    metadata = os.path.join(data_dir, 'HAM10000_metadata.csv')
    folders = [folder1, folder2]

    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/gradcam', exist_ok=True)

    print("Building dataframe...")
    df = build_dataframe(metadata, folders, args.mode)
    train_df, val_df, test_df = make_splits(df)
    print(f"Train {len(train_df)}  Val {len(val_df)}  Test {len(test_df)}")

    train_gen, val_gen, test_gen = create_gens(train_df, val_df, test_df, folders,
                                               img_size=(args.img_size,args.img_size),
                                               batch=args.batch, mode=args.mode)

    # Save class indices mapping for inference app
    class_indices = train_gen.class_indices  # label -> index
    with open('models/class_indices.json', 'w') as f:
        json.dump(class_indices, f)
    print("Saved models/class_indices.json:", class_indices)

    if args.mode == 'binary':
        num_classes = 2
    else:
        num_classes = len(class_indices)

    # Build model
    model, backbone = build_model_with_backbone(input_shape=(args.img_size, args.img_size, 3),
                                                num_classes=num_classes,
                                                base_trainable=False)
    model.summary()

    # Compute class weights (map label -> weight using train_df and class_indices)
    labels = train_df['label'].values
    classes_unique = np.array(sorted(train_gen.class_indices.keys(), key=lambda x: train_gen.class_indices[x]))
    # compute weights using label names -> convert to indices for compute_class_weight
    y_indices = np.array([train_gen.class_indices[l] for l in labels])
    unique_indices = np.unique(y_indices)
    cw = compute_class_weight(class_weight='balanced', classes=unique_indices, y=y_indices)
    # compute_class_weight returns in order of 'classes' argument; build dict mapping index->weight
    class_weight = {int(cls): float(w) for cls, w in zip(unique_indices, cw)}
    print("Using class_weight:", class_weight)

    # Callbacks
    monitor_metric = 'val_auc' if any('auc' in m.name for m in model.metrics) else 'val_loss'
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1),
        ModelCheckpoint('models/best_model.h5', save_best_only=True, monitor='val_loss')
    ]

    # Phase 1: train head
    print("Training head (backbone frozen)...")
    model.fit(train_gen, validation_data=val_gen, epochs=args.epochs, class_weight=class_weight, callbacks=callbacks)

    # Phase 2: fine-tune backbone last N layers
    N = args.fine_tune_layers
    print(f"Fine-tuning last {N} layers of backbone...")
    backbone.trainable = True
    total_backbone_layers = len(backbone.layers)
    freeze_up_to = max(0, total_backbone_layers - N)
    print(f"Total backbone layers: {total_backbone_layers}, freezing up to: {freeze_up_to}")
    for layer_idx, layer in enumerate(backbone.layers):
        layer.trainable = layer_idx >= freeze_up_to

    # Phase 2: fine-tune backbone last N layers
    # Phase 2: fine-tune backbone last N layers
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),loss=model.loss,metrics=model.metrics)

    # IMPORTANT: do NOT use class_weight during fine-tuning
    model.fit(train_gen,validation_data=val_gen,epochs=...,callbacks=callbacks)



    # Save final model
    model.save('models/cnn_ham10000_saved_model')
    print("Saved model to models/cnn_ham10000_saved_model")

    # --------------- Evaluation on test set ----------------
    steps = int(np.ceil(test_gen.samples / test_gen.batch_size))
    preds = model.predict(test_gen, steps=steps, verbose=1)
    if args.mode == 'binary':
        y_pred = (preds.ravel() >= 0.5).astype(int)
        idx2label = {0:'OTHER',1:'MEL'}
    else:
        y_pred = np.argmax(preds, axis=1)
        # invert class_indices to idx->label (sorted by index)
        idx2label_map = {v:k for k,v in train_gen.class_indices.items()}
        idx2label = [idx2label_map[i] for i in range(len(idx2label_map))]

    y_true = test_gen.classes

    # classification report
    target_names = idx2label if args.mode=='multiclass' else ['OTHER','MEL']
    print(classification_report(y_true, y_pred, target_names=target_names))

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, classes=target_names, out_path='results/confusion_matrix.png', normalize=False,
                          title='Confusion Matrix (counts)')
    plot_confusion_matrix(cm, classes=target_names, out_path='results/confusion_matrix_normalized.png', normalize=True,
                          title='Confusion Matrix (normalized)')

    # --------------- Accuracy & Loss Plots (using history from last fit if available) ---------------
    # Note: we used multiple .fit() calls; matplotlib plots from last history are helpful.
    # We will attempt to plot training history if available in callbacks; if not, user can track separately.
    # Here we try to load the 'models/best_model.h5' training history is not directly saved; skip.

    # --------------- Grad-CAM examples ---------------
    last_conv = find_last_conv_layer(model)
    if last_conv:
        print("Last conv layer for Grad-CAM:", last_conv)
        # pick up to 6 test images to visualize
        n_show = min(6, test_gen.samples)
        # get first n_show filenames from test_df
        test_filepaths = test_df['filepath'].values[:n_show]
        for i, fp in enumerate(test_filepaths):
            try:
                img_arr = kimage.img_to_array(kimage.load_img(fp, target_size=(args.img_size, args.img_size)))
                x = np.expand_dims(img_arr/255.0, axis=0)
                preds_raw = model.predict(x)
                pred_idx = int(np.argmax(preds_raw[0])) if args.mode=='multiclass' else int(preds_raw[0]>=0.5)
                heatmap = make_gradcam_heatmap(x, model, last_conv, pred_index=pred_idx)
                outp = f"results/gradcam/gradcam_{i}.png"
                save_gradcam_overlay(fp, outp, heatmap, alpha=0.4, target_size=(args.img_size, args.img_size))
                print("Saved Grad-CAM:", outp)
            except Exception as e:
                print("Grad-CAM failed for", fp, e)
    else:
        print("No convolutional layer found for Grad-CAM.")

    print("All done. Results saved in 'results/' and model saved in 'models/'.")


if __name__ == '__main__':
    main()
