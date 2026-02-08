"""
Data Loader for Semiconductor Defect Detection

Creates tf.data.Dataset pipelines for efficient training with:
- Parallel data loading
- Prefetching for GPU utilization  
- On-the-fly augmentation (optional)
- Grayscale to RGB conversion
"""

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from pathlib import Path
import numpy as np
import cv2
from typing import Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.config import (
    DATA_DIR, CLASS_NAMES, NUM_CLASSES,
    IMAGE_SIZE, BATCH_SIZE, AUGMENTATION_CONFIG
)


def parse_image(file_path: str, label: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Load and preprocess a single image.
    
    Args:
        file_path: Path to image file
        label: Class label (integer)
    
    Returns:
        Tuple of (image tensor, one-hot label)
    """
    # Read image
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=1)  # Grayscale
    
    # Convert grayscale to RGB (models expect 3 channels)
    img = tf.image.grayscale_to_rgb(img)
    
    # Resize if needed
    img = tf.image.resize(img, IMAGE_SIZE)
    
    # Convert to float32 [0, 1]
    img = tf.cast(img, tf.float32) / 255.0
    
    # One-hot encode label
    label = tf.one_hot(label, NUM_CLASSES)
    
    return img, label


def augment_image(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Apply random augmentations during training.
    
    Args:
        image: Input image tensor
        label: Label tensor (passed through)
    
    Returns:
        Augmented image and original label
    """
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    
    # Random vertical flip
    image = tf.image.random_flip_up_down(image)
    
    # Random rotation (approximate using crop + resize)
    # TF doesn't have direct rotation, so we use random crop
    image = tf.image.random_crop(image, size=[200, 200, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    
    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.1)
    
    # Random contrast
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    
    # Clip values to [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, label


def create_dataset(
    data_dir: Path,
    split: str = "train",
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
    augment: bool = False,
    cache: bool = True
) -> tf.data.Dataset:
    """
    Create tf.data.Dataset for training/validation/testing.
    
    Args:
        data_dir: Root data directory
        split: "train", "val", or "test"
        batch_size: Batch size
        shuffle: Whether to shuffle data
        augment: Whether to apply augmentation
        cache: Whether to cache dataset in memory
    
    Returns:
        tf.data.Dataset ready for model.fit()
    """
    split_dir = data_dir / split
    
    # Collect all image paths and labels
    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue
        
        for img_path in class_dir.glob("*.png"):
            image_paths.append(str(img_path))
            labels.append(class_idx)
    
    # Convert to tensors
    image_paths = tf.constant(image_paths)
    labels = tf.constant(labels, dtype=tf.int32)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    # Shuffle before caching for training
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)
    
    # Parse images in parallel
    dataset = dataset.map(
        parse_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Cache after parsing (before augmentation)
    if cache:
        dataset = dataset.cache()
    
    # Apply augmentation for training
    if augment:
        dataset = dataset.map(
            augment_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Batch
    dataset = dataset.batch(batch_size)
    
    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def get_class_weights(data_dir: Path, split: str = "train") -> dict:
    """
    Calculate class weights for imbalanced dataset.
    
    Args:
        data_dir: Root data directory
        split: Split to calculate weights for
    
    Returns:
        Dictionary mapping class index to weight
    """
    split_dir = data_dir / split
    
    class_counts = {}
    total = 0
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = split_dir / class_name
        if class_dir.exists():
            count = len(list(class_dir.glob("*.png")))
            class_counts[class_idx] = count
            total += count
    
    # Calculate weights (inverse frequency)
    n_classes = len(class_counts)
    weights = {}
    
    for class_idx, count in class_counts.items():
        if count > 0:
            weights[class_idx] = total / (n_classes * count)
        else:
            weights[class_idx] = 0.0
    
    return weights


def get_dataset_info(data_dir: Path) -> dict:
    """Get information about the dataset"""
    info = {"splits": {}}
    
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        
        split_info = {"classes": {}, "total": 0}
        
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if class_dir.exists():
                count = len(list(class_dir.glob("*.png")))
                split_info["classes"][class_name] = count
                split_info["total"] += count
        
        info["splits"][split] = split_info
    
    return info


if __name__ == "__main__":
    # Test data loading
    print("Testing data loader...")
    
    info = get_dataset_info(DATA_DIR)
    print("\nDataset Info:")
    for split, split_info in info["splits"].items():
        print(f"\n  {split}: {split_info['total']} images")
        for cls, count in split_info["classes"].items():
            if count > 0:
                print(f"    {cls}: {count}")
    
    # Test loading a batch
    print("\nLoading training batch...")
    train_ds = create_dataset(DATA_DIR, split="train", batch_size=8, augment=True)
    
    for images, labels in train_ds.take(1):
        print(f"  Batch shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Image range: [{images.numpy().min():.3f}, {images.numpy().max():.3f}]")
    
    print("\n✅ Data loader working!")
