"""
Ensemble Model for Semiconductor Defect Classification

Combines EfficientNet-B0 and MobileNetV3-Small predictions using
weighted averaging for improved accuracy while remaining edge-friendly.

Ensemble weights: EfficientNet 70%, MobileNet 30%
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.config import NUM_CLASSES, INPUT_SHAPE, ENSEMBLE_CONFIG


class EnsembleModel:
    """
    Weighted ensemble of EfficientNet and MobileNet models.
    
    Uses late fusion (prediction averaging) for edge-friendly deployment.
    """
    
    def __init__(
        self,
        efficientnet_model: Model,
        mobilenet_model: Model,
        efficientnet_weight: float = 0.7,
        mobilenet_weight: float = 0.3
    ):
        """
        Initialize ensemble with two models.
        
        Args:
            efficientnet_model: Trained EfficientNet model
            mobilenet_model: Trained MobileNet model
            efficientnet_weight: Weight for EfficientNet predictions
            mobilenet_weight: Weight for MobileNet predictions
        """
        self.efficientnet = efficientnet_model
        self.mobilenet = mobilenet_model
        self.eff_weight = efficientnet_weight
        self.mob_weight = mobilenet_weight
        
        # Normalize weights
        total_weight = self.eff_weight + self.mob_weight
        self.eff_weight /= total_weight
        self.mob_weight /= total_weight
        
        print(f"Ensemble weights: EfficientNet={self.eff_weight:.2f}, MobileNet={self.mob_weight:.2f}")
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            x: Input images (batch)
        
        Returns:
            Weighted average predictions
        """
        eff_preds = self.efficientnet.predict(x, verbose=0)
        mob_preds = self.mobilenet.predict(x, verbose=0)
        
        ensemble_preds = (
            self.eff_weight * eff_preds + 
            self.mob_weight * mob_preds
        )
        
        return ensemble_preds
    
    def predict_classes(self, x: np.ndarray) -> np.ndarray:
        """Get predicted class indices"""
        preds = self.predict(x)
        return np.argmax(preds, axis=1)
    
    def evaluate(self, x: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate ensemble on test data.
        
        Args:
            x: Test images
            y: True labels (one-hot or class indices)
        
        Returns:
            Dictionary with accuracy and per-class metrics
        """
        # Get predictions
        preds = self.predict(x)
        pred_classes = np.argmax(preds, axis=1)
        
        # Handle one-hot vs class indices
        if len(y.shape) > 1:
            true_classes = np.argmax(y, axis=1)
        else:
            true_classes = y
        
        # Calculate accuracy
        accuracy = np.mean(pred_classes == true_classes)
        
        # Per-class metrics
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(
            true_classes, pred_classes, average=None, zero_division=0
        )
        
        return {
            "accuracy": accuracy,
            "precision_per_class": precision,
            "recall_per_class": recall,
            "f1_per_class": f1,
            "support_per_class": support
        }


def create_functional_ensemble(
    efficientnet_model: Model,
    mobilenet_model: Model,
    efficientnet_weight: float = 0.7,
    mobilenet_weight: float = 0.3
) -> Model:
    """
    Create a functional Keras model that performs ensemble averaging.
    
    This can be saved and exported as a single model.
    
    Args:
        efficientnet_model: Trained EfficientNet
        mobilenet_model: Trained MobileNet
        efficientnet_weight: Weight for EfficientNet
        mobilenet_weight: Weight for MobileNet
    
    Returns:
        Keras Model performing weighted average
    """
    # Normalize weights
    total = efficientnet_weight + mobilenet_weight
    eff_w = efficientnet_weight / total
    mob_w = mobilenet_weight / total
    
    # Input
    inputs = layers.Input(shape=INPUT_SHAPE, name="input_image")
    
    # Get predictions from both models
    eff_pred = efficientnet_model(inputs)
    mob_pred = mobilenet_model(inputs)
    
    # Weighted average
    # Using Lambda layer for weighted sum
    def weighted_average(predictions):
        eff, mob = predictions
        return eff_w * eff + mob_w * mob
    
    outputs = layers.Lambda(
        weighted_average,
        name="ensemble_average"
    )([eff_pred, mob_pred])
    
    # Create ensemble model
    ensemble = Model(inputs=inputs, outputs=outputs, name="ensemble_classifier")
    
    return ensemble


if __name__ == "__main__":
    # Test imports
    print("Ensemble module loaded successfully")
    print(f"Default weights: EfficientNet={ENSEMBLE_CONFIG['efficientnet_weight']}, "
          f"MobileNet={ENSEMBLE_CONFIG['mobilenet_weight']}")
