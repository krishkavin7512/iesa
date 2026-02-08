"""
Simple Training Script - No Callbacks
To avoid JSON serialization errors in TF 2.10 on Windows
"""
import os
import sys
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

# Setup path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.config import *
from training.data_loader import create_dataset, get_class_weights
from models.efficientnet_model import create_efficientnet_model, unfreeze_layers

def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ Found {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")

def main():
    print("🚀 STARTING SIMPLE TRAINING (No Callbacks)")
    setup_gpu()
    
    # Data
    print("\n📂 Loading datasets...")
    train_ds = create_dataset(DATA_DIR, split="train", batch_size=BATCH_SIZE, augment=True)
    val_ds = create_dataset(DATA_DIR, split="val", batch_size=BATCH_SIZE, augment=False)
    
    weights = get_class_weights(DATA_DIR, split="train")
    weights = {int(k): float(v) for k, v in weights.items()}
    print(f"   Class weights: {len(weights)} classes")
    
    # Model
    print("\n🏗️ Creating EfficientNetB0...")
    model = create_efficientnet_model(freeze_base=True, freeze_percent=STAGE1_FREEZE_PERCENT)
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=STAGE1_LEARNING_RATE),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy']
    )
    
    # Stage 1
    print("\n📚 STAGE 1: Transfer Learning")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=STAGE1_EPOCHS,
        class_weight=weights,
        verbose=1
    )
    
    # Stage 2
    print("\n🔧 STAGE 2: Fine-tuning")
    model = unfreeze_layers(model, unfreeze_percent=STAGE2_UNFREEZE_PERCENT)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=STAGE2_LEARNING_RATE),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy']
    )
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=STAGE2_EPOCHS,
        class_weight=weights,
        verbose=1
    )
    
    # Save weights only to avoid JSON error
    print("\n💾 Saving model weights...")
    save_path = MODEL_DIR / "efficientnet_final_weights.h5"
    model.save_weights(str(save_path))
    print(f"   Saved: {save_path}")
    
    print("\n🎉 Training Complete!")

if __name__ == "__main__":
    main()
