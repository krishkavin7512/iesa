"""
MobileNetV3-Small Model for Semiconductor Defect Classification

Architecture:
- Base: MobileNetV3-Small (pretrained on ImageNet)
- Custom classification head with dropout and batch normalization
- Ultra-lightweight for edge deployment (~2.2MB → ~600KB after quantization)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras.applications import MobileNetV3Small
from pathlib import Path
import sys

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.config import (
    NUM_CLASSES, INPUT_SHAPE,
    MOBILENET_CONFIG, LABEL_SMOOTHING
)


def create_mobilenet_model(
    input_shape: tuple = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    dense_units: int = 96,
    dropout_1: float = 0.3,
    dropout_2: float = 0.2,
    l2_reg: float = 0.01,
    freeze_base: bool = True,
    freeze_percent: float = 0.75
) -> Model:
    """
    Create MobileNetV3-Small model with custom classification head.
    
    Args:
        input_shape: Input image shape (H, W, C)
        num_classes: Number of output classes
        dense_units: Units in dense layer (smaller than EfficientNet)
        dropout_1: First dropout rate
        dropout_2: Second dropout rate  
        l2_reg: L2 regularization factor
        freeze_base: Whether to freeze base model layers
        freeze_percent: Percentage of base layers to freeze
    
    Returns:
        Compiled Keras model
    """
    
    # Input layer
    inputs = layers.Input(shape=input_shape, name="input_image")
    
    # Base model - MobileNetV3Small
    base_model = MobileNetV3Small(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling=None,
        include_preprocessing=True
    )
    base_model._name = "mobilenet_base"
    
    # Freeze layers if specified
    if freeze_base:
        total_layers = len(base_model.layers)
        freeze_until = int(total_layers * freeze_percent)
        
        for i, layer in enumerate(base_model.layers):
            if i < freeze_until:
                layer.trainable = False
            else:
                layer.trainable = True
        
        print(f"MobileNet: Freezing {freeze_until}/{total_layers} layers ({freeze_percent*100:.0f}%)")
    
    # Pass input through base model
    x = base_model(inputs)
    
    # Custom classification head
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    
    # Dropout 1
    x = layers.Dropout(dropout_1, name="dropout_1")(x)
    
    # Dense layer with L2 regularization
    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="dense_1"
    )(x)
    
    # Batch normalization
    x = layers.BatchNormalization(name="batch_norm")(x)
    
    # Dropout 2
    x = layers.Dropout(dropout_2, name="dropout_2")(x)
    
    # Output layer
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions"
    )(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name="mobilenet_defect_classifier")
    
    return model


def compile_model(model: Model, learning_rate: float = 1e-3) -> Model:
    """
    Compile model with optimizer, loss, and metrics.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled model
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    loss = keras.losses.CategoricalCrossentropy(
        label_smoothing=LABEL_SMOOTHING
    )
    
    metrics = [
        keras.metrics.CategoricalAccuracy(name="accuracy"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc"),
    ]
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model


def unfreeze_layers(model: Model, unfreeze_percent: float = 0.25) -> Model:
    """
    Unfreeze last N% of base model layers for fine-tuning.
    
    Args:
        model: Compiled model
        unfreeze_percent: Percentage of layers to unfreeze
    
    Returns:
        Model with unfrozen layers (needs recompilation)
    """
    # Find the base model within our model
    base_model = None
    for layer in model.layers:
        if "mobilenet" in layer.name.lower():
            base_model = layer
            break
    
    if base_model is None:
        print("Warning: Could not find MobileNet base model")
        total_layers = len(model.layers)
        unfreeze_from = int(total_layers * (1 - unfreeze_percent))
        for i, layer in enumerate(model.layers):
            if i >= unfreeze_from:
                layer.trainable = True
        return model
    
    # Unfreeze last N% of base model layers
    total_layers = len(base_model.layers)
    unfreeze_from = int(total_layers * (1 - unfreeze_percent))
    
    for i, layer in enumerate(base_model.layers):
        if i >= unfreeze_from:
            layer.trainable = True
    
    trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
    print(f"MobileNet: {trainable_count}/{total_layers} layers now trainable")
    
    return model


def get_model_summary(model: Model) -> dict:
    """Get model summary statistics"""
    trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.non_trainable_weights
    )
    total_params = trainable_params + non_trainable_params
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "estimated_size_mb": total_params * 4 / (1024 * 1024)  # FP32
    }


if __name__ == "__main__":
    # Test model creation
    print("Creating MobileNetV3-Small model...")
    
    model = create_mobilenet_model(
        freeze_base=True,
        freeze_percent=0.75
    )
    
    model = compile_model(model, learning_rate=1e-3)
    
    model.summary()
    
    stats = get_model_summary(model)
    print(f"\nModel Statistics:")
    print(f"  Total params: {stats['total_params']:,}")
    print(f"  Trainable: {stats['trainable_params']:,}")
    print(f"  Non-trainable: {stats['non_trainable_params']:,}")
    print(f"  Estimated size: {stats['estimated_size_mb']:.2f} MB")
