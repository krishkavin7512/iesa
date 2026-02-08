"""
Quick Training Script - CPU Optimized

Uses a smaller subset and fewer epochs for fast iteration on CPU.
For full training, use the Colab notebook or GPU machine.

Run: python training/train_quick.py
"""

import os
import sys
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras.applications import MobileNetV3Small  # Smaller model for CPU
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "augmented"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "clean", "scratches", "particles", "pattern_defects",
    "edge_defects", "center_defects", "random_defects", "other"
]
NUM_CLASSES = 8
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)

# Quick training settings for CPU
BATCH_SIZE = 16  # Smaller batch for CPU
EPOCHS = 5       # Quick epochs
LR = 1e-3


def create_dataset(split, batch_size=16, max_samples=1000, augment=False):
    """Create dataset with limited samples for quick training"""
    split_dir = DATA_DIR / split
    
    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue
        imgs = list(class_dir.glob("*.png"))[:max_samples // NUM_CLASSES]
        for img_path in imgs:
            image_paths.append(str(img_path))
            labels.append(class_idx)
    
    def parse_fn(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.grayscale_to_rgb(img)
        img = tf.image.resize(img, IMAGE_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img, tf.one_hot(label, NUM_CLASSES)
    
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.shuffle(len(image_paths))
    ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, len(image_paths)


def create_mobilenet():
    """Create lightweight MobileNetV3-Small"""
    inputs = layers.Input(shape=INPUT_SHAPE)
    
    base = MobileNetV3Small(
        include_top=False,
        weights="imagenet",
        input_shape=INPUT_SHAPE,
        include_preprocessing=True
    )
    
    # Freeze most layers
    for layer in base.layers[:-20]:
        layer.trainable = False
    
    x = base(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    return Model(inputs, outputs, name="mobilenet_quick")


def main():
    print("=" * 60)
    print("🚀 QUICK TRAINING (CPU Optimized)")
    print("=" * 60)
    
    # Load limited data
    print("\n📂 Loading data (limited samples)...")
    train_ds, train_n = create_dataset("train", BATCH_SIZE, max_samples=800)
    val_ds, val_n = create_dataset("val", BATCH_SIZE, max_samples=200)
    print(f"   Train: {train_n} samples, Val: {val_n} samples")
    
    # Create model
    print("\n🏗️ Creating MobileNetV3-Small...")
    model = create_mobilenet()
    
    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    # Quick summary
    trainable = sum(layer.count_params() for layer in model.layers if layer.trainable)
    print(f"   Trainable params: {trainable:,}")
    
    # Train
    print("\n📚 Training...")
    callbacks = [
        keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            str(OUTPUT_DIR / "mobilenet_quick.keras"),
            save_best_only=True
        )
    ]
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Report results
    best_acc = max(history.history.get('val_accuracy', [0]))
    print(f"\n✅ Best validation accuracy: {best_acc:.4f}")
    
    # Save final model
    model.save(str(OUTPUT_DIR / "mobilenet_quick_final.keras"))
    model.save(str(OUTPUT_DIR / "mobilenet_quick_final.h5"))
    print(f"\n💾 Models saved to: {OUTPUT_DIR}")
    
    print("\n🎉 Quick training complete!")
    print("For full training with GPU, use the Colab notebook: notebooks/training_colab.ipynb")


if __name__ == "__main__":
    main()
