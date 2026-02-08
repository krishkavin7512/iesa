"""
Training Script for Semiconductor Defect Detection

Two-stage training approach:
1. Stage 1 (Transfer Learning): Freeze base, train classification head
2. Stage 2 (Fine-tuning): Unfreeze top layers, lower learning rate

Usage:
    python training/train.py                    # Train EfficientNet
    python training/train.py --model mobilenet  # Train MobileNet
    python training/train.py --model both       # Train both models
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.config import (
    DATA_DIR, MODEL_DIR, LOG_DIR, CHECKPOINT_DIR,
    NUM_CLASSES, CLASS_NAMES, BATCH_SIZE,
    STAGE1_EPOCHS, STAGE1_LEARNING_RATE, STAGE1_FREEZE_PERCENT,
    STAGE2_EPOCHS, STAGE2_LEARNING_RATE, STAGE2_UNFREEZE_PERCENT,
    EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR,
    TENSORBOARD_ENABLED, SAVE_BEST_ONLY, MONITOR_METRIC, MONITOR_MODE
)
from training.data_loader import create_dataset, get_class_weights, get_dataset_info
from models.efficientnet_model import (
    create_efficientnet_model, 
    compile_model as compile_efficientnet,
    unfreeze_layers as unfreeze_efficientnet,
    get_model_summary
)
from models.mobilenet_model import (
    create_mobilenet_model,
    compile_model as compile_mobilenet,
    unfreeze_layers as unfreeze_mobilenet
)


def setup_gpu():
    """Configure GPU for training"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ Found {len(gpus)} GPU(s)")
            for gpu in gpus:
                print(f"   - {gpu.name}")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("⚠️ No GPU found, using CPU")


def get_callbacks(model_name: str, stage: int) -> list:
    """
    Create training callbacks.
    
    Args:
        model_name: Name of model for logging
        stage: Training stage (1 or 2)
    
    Returns:
        List of Keras callbacks
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = []
    
    # TensorBoard
    if TENSORBOARD_ENABLED:
        tb_dir = LOG_DIR / f"{model_name}_stage{stage}_{timestamp}"
        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=str(tb_dir),
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                update_freq='epoch'
            )
        )
    
    # Model checkpoint
    # ckpt_path = CHECKPOINT_DIR / f"{model_name}_stage{stage}_best.h5"
    # callbacks.append(
    #     keras.callbacks.ModelCheckpoint(
    #         filepath=str(ckpt_path),
    #         monitor=MONITOR_METRIC,
    #         mode=MONITOR_MODE,
    #         save_best_only=SAVE_BEST_ONLY,
    #         verbose=1
    #     )
    # )
    
    # Early stopping
    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor=MONITOR_METRIC,
            mode=MONITOR_MODE,
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
    )
    
    # Reduce learning rate on plateau
    callbacks.append(
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        )
    )
    
    # CSV logger (Disabled to avoid JSON serialization errors with EagerTensor)
    # csv_path = LOG_DIR / f"{model_name}_stage{stage}_{timestamp}.csv"
    # callbacks.append(
    #     keras.callbacks.CSVLogger(str(csv_path))
    # )
    
    return callbacks


def train_model(
    model_type: str = "efficientnet",
    batch_size: int = BATCH_SIZE,
    stage1_epochs: int = STAGE1_EPOCHS,
    stage2_epochs: int = STAGE2_EPOCHS
):
    """
    Train a model using two-stage approach.
    
    Args:
        model_type: "efficientnet" or "mobilenet"
        batch_size: Training batch size
        stage1_epochs: Epochs for stage 1
        stage2_epochs: Epochs for stage 2
    
    Returns:
        Trained model and training history
    """
    print("\n" + "=" * 70)
    print(f"🚀 TRAINING {model_type.upper()} MODEL")
    print("=" * 70)
    
    # Setup
    setup_gpu()
    
    # Load datasets
    print("\n📂 Loading datasets...")
    train_ds = create_dataset(DATA_DIR, split="train", batch_size=batch_size, augment=True)
    val_ds = create_dataset(DATA_DIR, split="val", batch_size=batch_size, augment=False)
    
    # Get class weights for imbalanced data
    class_weights = get_class_weights(DATA_DIR, split="train")
    # Cast to pure python types to avoid JSON serialization errors
    class_weights = {int(k): float(v) for k, v in class_weights.items()}
    print(f"   Class weights calculated for {len(class_weights)} classes")
    
    # Create model
    print(f"\n🏗️ Creating {model_type} model...")
    
    if model_type == "efficientnet":
        model = create_efficientnet_model(
            freeze_base=True,
            freeze_percent=STAGE1_FREEZE_PERCENT
        )
        compile_fn = compile_efficientnet
        unfreeze_fn = unfreeze_efficientnet
    else:  # mobilenet
        model = create_mobilenet_model(
            freeze_base=True,
            freeze_percent=0.75
        )
        compile_fn = compile_mobilenet
        unfreeze_fn = unfreeze_mobilenet
    
    # Print model info
    stats = get_model_summary(model)
    print(f"   Total params: {stats['total_params']:,}")
    print(f"   Trainable: {stats['trainable_params']:,}")
    print(f"   Size: ~{stats['estimated_size_mb']:.1f} MB")
    
    # =========================================================================
    # STAGE 1: Transfer Learning
    # =========================================================================
    print("\n" + "-" * 70)
    print("📚 STAGE 1: Transfer Learning (frozen base)")
    print("-" * 70)
    
    model = compile_fn(model, learning_rate=STAGE1_LEARNING_RATE)
    
    callbacks = get_callbacks(model_type, stage=1)
    
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage1_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Get best stage 1 metrics
    best_acc = max(history1.history.get('val_accuracy', [0]))
    print(f"\n✅ Stage 1 complete - Best val_accuracy: {best_acc:.4f}")
    
    # =========================================================================
    # STAGE 2: Fine-tuning
    # =========================================================================
    print("\n" + "-" * 70)
    print("🔧 STAGE 2: Fine-tuning (unfrozen top layers)")
    print("-" * 70)
    
    # Unfreeze top layers
    model = unfreeze_fn(model, unfreeze_percent=STAGE2_UNFREEZE_PERCENT)
    
    # Recompile with lower learning rate
    model = compile_fn(model, learning_rate=STAGE2_LEARNING_RATE)
    
    callbacks = get_callbacks(model_type, stage=2)
    
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage2_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Get final metrics
    best_acc = max(history2.history.get('val_accuracy', [0]))
    print(f"\n✅ Stage 2 complete - Best val_accuracy: {best_acc:.4f}")
    
    # =========================================================================
    # Save final model
    # =========================================================================
    print("\n💾 Saving final model...")
    
    # Save in Keras format
    # Save final model
    final_path = MODEL_DIR / f"{model_type}_final.h5"
    model.save(str(final_path))
    print(f"   Saved: {final_path}")
    
    # Also save in H5 format
    h5_path = MODEL_DIR / f"{model_type}_final.h5"
    model.save(str(h5_path))
    print(f"   Saved: {h5_path}")
    
    # Combine histories
    full_history = {
        'stage1': history1.history,
        'stage2': history2.history
    }
    
    return model, full_history


def train_both_models():
    """Train both EfficientNet and MobileNet models"""
    
    results = {}
    
    # Train EfficientNet
    eff_model, eff_history = train_model("efficientnet")
    results["efficientnet"] = {
        "model": eff_model,
        "history": eff_history
    }
    
    # Train MobileNet
    mob_model, mob_history = train_model("mobilenet")
    results["mobilenet"] = {
        "model": mob_model,
        "history": mob_history
    }
    
    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train defect detection models")
    parser.add_argument(
        "--model", 
        type=str, 
        default="efficientnet",
        choices=["efficientnet", "mobilenet", "both"],
        help="Model to train"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Training batch size"
    )
    parser.add_argument(
        "--stage1-epochs",
        type=int,
        default=STAGE1_EPOCHS,
        help="Stage 1 epochs"
    )
    parser.add_argument(
        "--stage2-epochs",
        type=int,
        default=STAGE2_EPOCHS,
        help="Stage 2 epochs"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔬 SEMICONDUCTOR DEFECT DETECTION - MODEL TRAINING")
    print("=" * 70)
    
    # Print dataset info
    print("\n📊 Dataset Overview:")
    info = get_dataset_info(DATA_DIR)
    for split, split_info in info["splits"].items():
        print(f"   {split}: {split_info['total']} images")
    
    if args.model == "both":
        results = train_both_models()
        print("\n" + "=" * 70)
        print("🎉 TRAINING COMPLETE - BOTH MODELS")
        print("=" * 70)
    else:
        model, history = train_model(
            model_type=args.model,
            batch_size=args.batch_size,
            stage1_epochs=args.stage1_epochs,
            stage2_epochs=args.stage2_epochs
        )
        print("\n" + "=" * 70)
        print(f"🎉 TRAINING COMPLETE - {args.model.upper()}")
        print("=" * 70)
    
    print(f"\n📍 Models saved to: {MODEL_DIR}")
    print(f"📍 Logs saved to: {LOG_DIR}")
    print("\n🔜 Next step: python evaluation/evaluate.py")


if __name__ == "__main__":
    main()
