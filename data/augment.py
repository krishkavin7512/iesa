"""
Data Augmentation Script for Semiconductor Defect Detection

Expands the curated dataset from ~1,500 to ~4,000+ images using:
- Geometric transforms (rotation, flips, shifts)
- Intensity transforms (brightness, contrast, noise)
- Advanced transforms (elastic, grid distortion, cutout)

Also creates train/val/test splits with stratification.

Run: python data/augment.py
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from sklearn.model_selection import train_test_split
from collections import defaultdict
import shutil

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
AUGMENTED_DATA_DIR = PROJECT_ROOT / "data" / "augmented"

# Target image size
TARGET_SIZE = (224, 224)

# Class names
CLASS_NAMES = [
    "clean",
    "scratches",
    "particles", 
    "pattern_defects",
    "edge_defects",
    "center_defects",
    "random_defects",
    "other"
]

# Target images per class (for balancing)
TARGET_PER_CLASS_TRAIN = 1500  # Balanced target for all classes
TARGET_PER_CLASS_VAL = 75
TARGET_PER_CLASS_TEST = 75


def get_augmentation_pipeline():
    """
    Create augmentation pipeline using Albumentations
    
    Transforms designed for semiconductor defect images:
    - Preserve defect characteristics
    - Simulate real-world variations
    - Avoid unrealistic distortions
    """
    return A.Compose([
        # Geometric transforms
        A.OneOf([
            A.Rotate(limit=20, border_mode=cv2.BORDER_REFLECT, p=0.7),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        ], p=0.6),
        
        A.OneOf([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ], p=0.5),
        
        # Intensity transforms
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),
        ], p=0.5),
        
        # Noise and blur (simulating sensor variations)
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.MedianBlur(blur_limit=3, p=0.2),
        ], p=0.3),
        
        # Advanced transforms
        A.OneOf([
            A.ElasticTransform(alpha=50, sigma=5, p=0.3),
            A.GridDistortion(num_steps=3, distort_limit=0.1, p=0.3),
        ], p=0.2),
        
        # Cutout (simulates occlusions/noise)
        A.CoarseDropout(
            max_holes=8, 
            max_height=20, 
            max_width=20,
            min_holes=1,
            min_height=5,
            min_width=5,
            fill_value=128,
            p=0.2
        ),
    ])


def get_light_augmentation():
    """Light augmentation for validation/test (only geometric)"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ])


def augment_image(image: np.ndarray, transform: A.Compose, num_augmentations: int = 1):
    """Generate augmented versions of an image"""
    augmented = []
    for _ in range(num_augmentations):
        result = transform(image=image)
        augmented.append(result['image'])
    return augmented


def load_processed_images():
    """Load all images from processed directory"""
    all_images = []
    
    for class_name in CLASS_NAMES:
        class_dir = PROCESSED_DATA_DIR / class_name
        if not class_dir.exists():
            print(f"⚠️ Class directory not found: {class_dir}")
            continue
            
        images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
        for img_path in images:
            all_images.append({
                "path": img_path,
                "class": class_name
            })
    
    return all_images


def create_splits(images: list, test_size: float = 0.15, val_size: float = 0.15):
    """Create stratified train/val/test splits"""
    
    # Group by class
    class_images = defaultdict(list)
    for img in images:
        class_images[img['class']].append(img)
    
    train_data = []
    val_data = []
    test_data = []
    
    for class_name, class_imgs in class_images.items():
        n = len(class_imgs)
        
        if n < 10:
            print(f"⚠️ Too few images for {class_name}: {n}")
            # Put all in train
            train_data.extend(class_imgs)
            continue
        
        # Split
        train_val, test = train_test_split(
            class_imgs, 
            test_size=test_size, 
            random_state=42
        )
        
        # Adjust val_size for remaining data
        adjusted_val_size = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val, 
            test_size=adjusted_val_size, 
            random_state=42
        )
        
        train_data.extend(train)
        val_data.extend(val)
        test_data.extend(test)
    
    return train_data, val_data, test_data


def augment_and_save_split(
    data: list, 
    split_name: str, 
    target_per_class: int,
    transform: A.Compose,
    augment_factor: int = 3
):
    """Augment and save images for a split"""
    
    print(f"\n📁 Processing {split_name} split...")
    
    # Group by class
    class_images = defaultdict(list)
    import random
    for img in data:
        class_images[img['class']].append(img)
    
    saved_counts = defaultdict(int)
    
    for class_name in CLASS_NAMES:
        class_imgs = class_images.get(class_name, [])
        if not class_imgs:
            print(f"  ⚠️ No images for {class_name}")
            continue
        
        # Shuffle to ensure random sampling for undersampling
        random.shuffle(class_imgs)

        target_dir = AUGMENTED_DATA_DIR / split_name / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear existing
        for f in target_dir.glob("*"):
            f.unlink()
        
        # Calculate how many augmentations needed
        current_count = len(class_imgs)
        
        # If we have too many, we undersample (just take the first N after shuffle)
        # If we have too few, we oversample
        
        # For training, we force balance to target_per_class
        # For val/test, we usually keep original distribution but apply light augs?
        # Actually, user wants to balance training. Val/Test should represent real world?
        # Standard practice: Balance TRAIN, keep VAL/TEST representative (imbalanced).
        # But 'target_per_class' is passed in.
        # In main(), val/test targets are 75. 
        # If val has > 75 (e.g. Pattern), we should probably KEEP them for robust eval.
        # But wait, create_splits makes stratified splits.
        # Imbalance in original -> Imbalance in splits.
        # If we want balanced training, we enforce target_per_class.
        # If we want representative val/test, we should NOT cap them?
        # The current code enforces 'needed = max(target, current)'.
        # I want to change this ONLY for TRAIN to Enforce Balance (Undersample).
        
        if split_name == "train":
             # Strict balancing: Undersample if > target, Oversample if < target
             needed = target_per_class 
             # If current > target, we only use first 'target' images.
             effective_imgs = class_imgs[:target_per_class] if current_count > target_per_class else class_imgs
             current_count = len(effective_imgs)
             
             augs_per_image = max(1, int(np.ceil(target_per_class / current_count)))
        else:
             # Validation/Test: Keep all originals, no extensive oversampling unless very small?
             # Current logic was: target=75.
             # If we have 1000 pattern defects in val?
             # keep them.
             effective_imgs = class_imgs
             needed = max(target_per_class, current_count)
             current_count = len(effective_imgs)
             augs_per_image = 1 # No heavy augmentation for val/test usually
        
        saved = 0
        img_idx = 0
        
        # Loop until we reach target or exhaust images
        # We might need to loop multiple times over images if oversampling
        
        pbar = tqdm(total=needed, desc=f"  {class_name}")
        
        while saved < needed:
            # Get next image (cyclical if oversampling)
            img_info = effective_imgs[img_idx % len(effective_imgs)]
            img_idx += 1
            
            try:
                # Load image
                img = cv2.imread(str(img_info['path']), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                # If we haven't saved the original yet (first pass), save it
                if img_idx <= len(effective_imgs):
                     save_path = target_dir / f"{class_name}_{saved:05d}.png"
                     cv2.imwrite(str(save_path), img)
                     saved += 1
                     pbar.update(1)
                else:
                    # Second pass+ (Augmentation)
                    if split_name == "train":
                         aug_imgs = augment_image(img, transform, 1)
                         for aug_img in aug_imgs:
                             if saved >= needed: break
                             save_path = target_dir / f"{class_name}_{saved:05d}.png"
                             cv2.imwrite(str(save_path), aug_img)
                             saved += 1
                             pbar.update(1)
                    else:
                        # Val/Test: If we need more than we have (unlikely for majority, likely for minority)
                        # We just save original again? Or light augment?
                        # Using transform (light_transform for val/test)
                        aug_imgs = augment_image(img, transform, 1)
                        for aug_img in aug_imgs:
                             if saved >= needed: break
                             save_path = target_dir / f"{class_name}_{saved:05d}.png"
                             cv2.imwrite(str(save_path), aug_img)
                             saved += 1
                             pbar.update(1)
                             
            except Exception as e:
                print(f"Error: {e}")
                continue
                
            if img_idx > len(effective_imgs) * 10 and saved < needed:
                 # Safety break for infinite loops if augmentation fails
                 print("Warning: Augmentation loop safety break")
                 break
        
        pbar.close()
        saved_counts[class_name] = saved
    
    return saved_counts


def print_split_stats(split_name: str):
    """Print statistics for a split"""
    split_dir = AUGMENTED_DATA_DIR / split_name
    
    print(f"\n📊 {split_name.upper()} Split Statistics:")
    total = 0
    for class_name in CLASS_NAMES:
        class_dir = split_dir / class_name
        if class_dir.exists():
            count = len(list(class_dir.glob("*.png")))
            bar = "█" * (count // 20)
            print(f"    {class_name:20s}: {count:4d} {bar}")
            total += count
    print(f"    {'TOTAL':20s}: {total:4d}")
    return total


def main():
    """Main augmentation orchestration"""
    print("=" * 60)
    print("🔬 SEMICONDUCTOR DEFECT DETECTION - DATA AUGMENTATION")
    print("=" * 60)
    
    # Create directories
    for split in ["train", "val", "test"]:
        for cls in CLASS_NAMES:
            (AUGMENTED_DATA_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    
    # Load processed images
    print("\n📂 Loading processed images...")
    images = load_processed_images()
    print(f"   Found {len(images)} images")
    
    if len(images) == 0:
        print("❌ No images found! Run data curation first:")
        print("   python data/curate_datasets.py")
        return
    
    # Print current distribution
    class_counts = defaultdict(int)
    for img in images:
        class_counts[img['class']] += 1
    
    print("\n📊 Original Distribution:")
    for cls in CLASS_NAMES:
        count = class_counts.get(cls, 0)
        print(f"    {cls:20s}: {count:4d}")
    
    # Create splits
    print("\n✂️ Creating train/val/test splits...")
    train_data, val_data, test_data = create_splits(images)
    print(f"   Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Get augmentation pipelines
    train_transform = get_augmentation_pipeline()
    light_transform = get_light_augmentation()
    
    # Augment and save each split
    augment_and_save_split(train_data, "train", TARGET_PER_CLASS_TRAIN, train_transform)
    augment_and_save_split(val_data, "val", TARGET_PER_CLASS_VAL, light_transform)
    augment_and_save_split(test_data, "test", TARGET_PER_CLASS_TEST, light_transform)
    
    # Print final statistics
    print("\n" + "=" * 60)
    print("📊 FINAL DATASET STATISTICS")
    print("=" * 60)
    
    train_total = print_split_stats("train")
    val_total = print_split_stats("val")
    test_total = print_split_stats("test")
    
    grand_total = train_total + val_total + test_total
    print(f"\n🎯 GRAND TOTAL: {grand_total} images")
    print(f"   Train: {train_total} ({100*train_total/grand_total:.1f}%)")
    print(f"   Val:   {val_total} ({100*val_total/grand_total:.1f}%)")
    print(f"   Test:  {test_total} ({100*test_total/grand_total:.1f}%)")
    
    # Create metadata
    metadata = []
    for split in ["train", "val", "test"]:
        split_dir = AUGMENTED_DATA_DIR / split
        for cls in CLASS_NAMES:
            class_dir = split_dir / cls
            if class_dir.exists():
                for img_path in class_dir.glob("*.png"):
                    metadata.append({
                        "filename": img_path.name,
                        "split": split,
                        "class": cls,
                        "path": str(img_path)
                    })
    
    metadata_df = pd.DataFrame(metadata)
    metadata_path = AUGMENTED_DATA_DIR / "metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    print(f"\n💾 Metadata saved: {metadata_path}")
    
    print("\n🎉 Augmentation complete! Next step:")
    print("   python training/train.py")


if __name__ == "__main__":
    main()
