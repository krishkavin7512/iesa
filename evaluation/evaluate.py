"""
Evaluation Script for Semiconductor Defect Detection

Generates comprehensive metrics and visualizations:
1. Confusion Matrix
2. Classification Report
3. ROC Curves (per class)
4. Misclassification Analysis
5. Inference Latency Test
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_fscore_support
)
import time
import json

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import config
from training.config import (
    DATA_DIR, MODEL_DIR, OUTPUT_DIR,
    CLASS_NAMES, BATCH_SIZE, INPUT_SHAPE
)
from training.data_loader import create_dataset
from models.efficientnet_model import create_efficientnet_model

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("talk")

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"   Saved: {save_path}")

def plot_roc_curves(y_true_onehot, y_pred_probs, save_path):
    """Plot ROC curves for each class"""
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    n_classes = len(CLASS_NAMES)
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i], tpr[i], 
            color=color, 
            lw=2,
            label=f'{CLASS_NAMES[i]} (AUC = {roc_auc[i]:.2f})'
        )
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves by Class')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"   Saved: {save_path}")

def evaluate_model(model_path, split='test', model_type=None):
    """Run full evaluation pipeline"""
    print(f"\n🚀 EVALUATING MODEL: {model_path.name}")
    print("=" * 60)
    
    # Load model
    print("Loading model...")
    try:
        if model_type:
            # Rebuild model and load weights
            if model_type == "efficientnet":
                model = create_efficientnet_model()
            elif model_type == "mobilenet":
                # Import here to avoid circular dependency if not needed
                from models.mobilenet_model import create_mobilenet_model
                model = create_mobilenet_model()
            else:
                 raise ValueError(f"Unknown model type: {model_type}")
            
            # compile to avoid warnings
            model.compile(metrics=['accuracy'])
            model.load_weights(str(model_path))
            print("   Loaded weights successfully.")
        else:
            model = keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    # Load dataset
    print(f"Loading {split} dataset...")
    # Note: augment=False, shuffle=False for evaluation
    ds = create_dataset(DATA_DIR, split=split, batch_size=BATCH_SIZE, augment=False, shuffle=False)
    
    # Get labels and predictions
    print("Running inference...")
    y_true = []
    y_pred_probs = []
    
    # We need to iterate carefully to keep order matching (shuffle=False in create_dataset logic needed)
    # My create_dataset shuffles by default unless modified?
    # Let's verify create_dataset logic
    # In training/data_loader.py, if split != 'train', it usually doesn't shuffle?
    # Wait, create_dataset calls .shuffle(len(paths)).
    # Check data_loader.py: 
    # ds = ds.shuffle(len(image_paths)) # ALWAYS shuffles!
    # This is BAD for ordered evaluation if we want to map back to filenames.
    # However, create_dataset returns (image, label).
    # So we can just iterate.
    
    for images, labels in ds:
        probs = model.predict(images, verbose=0)
        y_pred_probs.extend(probs)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Metrics
    print("\n📊 Generating Metrics...")
    
    # 1. Classification Report
    labels = list(range(len(CLASS_NAMES)))
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, labels=labels, output_dict=True)
    print("\n" + classification_report(y_true, y_pred, target_names=CLASS_NAMES, labels=labels))
    
    # Save metrics
    metrics_dir = OUTPUT_DIR / f"metrics_{model_path.stem}"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    # 2. Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, metrics_dir / "confusion_matrix.png")
    
    # 3. ROC Curves
    # Need one-hot y_true
    y_true_onehot = tf.keras.utils.to_categorical(y_true, len(CLASS_NAMES))
    plot_roc_curves(y_true_onehot, y_pred_probs, metrics_dir / "roc_curves.png")
    
    # 4. Latency
    print("\n⏱️ Measuring Latency...")
    dummy_input = tf.random.normal((1, *INPUT_SHAPE))
    # Warmup
    for _ in range(10):
        _ = model(dummy_input)
        
    start_time = time.time()
    for _ in range(100):
        _ = model(dummy_input)
    end_time = time.time()
    
    avg_latency = (end_time - start_time) * 10  # ms per sample
    print(f"   Average Latency: {avg_latency:.2f} ms")
    
    # 5. Model Size
    model_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"   Model Size: {model_size:.2f} MB")
    
    print(f"\n✅ Evaluation complete! Results saved in: {metrics_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .h5 or .keras model file")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on")
    parser.add_argument("--model-type", type=str, default=None, choices=["efficientnet", "mobilenet"], help="Model type for weight loading")
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        # Try looking in MODEL_DIR
        model_path = MODEL_DIR / args.model
        if not model_path.exists():
             print(f"Error: Model file not found: {args.model}")
             sys.exit(1)
             
    evaluate_model(model_path, args.split, args.model_type)

if __name__ == "__main__":
    main()
