"""
Dataset Curation Script v2 - Optimized for Downloaded Datasets

Maps downloaded raw datasets to our 8 target classes:
1. Clean - No defects
2. Scratches - Linear defect patterns  
3. Particles - Spot/particle contamination
4. Pattern_Defects - Bridges, shorts, pattern issues
5. Edge_Defects - Defects at wafer edges
6. Center_Defects - Defects at wafer center
7. Random_Defects - Random scattered defects
8. Other - Ambiguous or multiple defects

Supports:
- WaferMap (balanced/imbalanced folders with labeled images)
- DeepPCB (PCB defect images)
- WM811K (pickle file with wafer maps - requires conversion)

Run: python data/curate_datasets.py
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import shutil
from collections import defaultdict
import pickle

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Target image size
TARGET_SIZE = (224, 224)

# Class mapping configuration
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


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for consistency:
    1. Convert to grayscale if needed
    2. Resize to target size
    3. Apply CLAHE for contrast enhancement
    """
    # Convert to grayscale if color
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize
    image = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    
    return image


def curate_wafermap():
    """
    Curate WaferMap dataset (shawon10/wafermap from Kaggle)
    
    Structure: WaferMap/balanced/{Center, Donut, Edge-loc, Edge-ring, Loc, Near-Full, None, Random, Scratch}
    
    Mapping to our classes:
    - None → clean
    - Scratch → scratches
    - Edge-loc, Edge-ring → edge_defects
    - Center, Donut → center_defects
    - Random → random_defects
    - Loc, Near-Full → pattern_defects
    """
    print("\n" + "=" * 60)
    print("🔧 Curating WaferMap Dataset...")
    print("=" * 60)
    
    source_dir = RAW_DATA_DIR / "wafermap" / "WaferMap" / "balanced"
    if not source_dir.exists():
        source_dir = RAW_DATA_DIR / "wafermap" / "WaferMap" / "imbalanced"
    if not source_dir.exists():
        print("❌ WaferMap not found. Skipping...")
        return []
    
    # Class mapping - normalize folder names
    class_mapping = {
        "none": "clean",
        "scratch": "scratches",
        "edge-loc": "edge_defects",
        "edge-ring": "edge_defects",
        "edgeloc": "edge_defects",
        "edgering": "edge_defects",
        "center": "center_defects",
        "donut": "center_defects",
        "random": "random_defects",
        "loc": "pattern_defects",
        "near-full": "pattern_defects",
        "nearfull": "pattern_defects",
    }
    
    curated = []
    
    for class_dir in source_dir.iterdir():
        if not class_dir.is_dir():
            continue
            
        original_class = class_dir.name.lower().replace("_", "-").replace(" ", "")
        target_class = None
        
        # Fuzzy match class name
        for key, value in class_mapping.items():
            if key in original_class or original_class in key:
                target_class = value
                break
        
        if target_class is None:
            print(f"  ⚠️ Unknown class: {class_dir.name}")
            continue
            
        target_dir = PROCESSED_DATA_DIR / target_class
        target_dir.mkdir(parents=True, exist_ok=True)
        
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        
        for img_path in tqdm(images, desc=f"WaferMap/{class_dir.name}"):
            try:
                # Load and preprocess
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                    
                img = preprocess_image(img)
                
                # Save to processed folder
                new_name = f"wmap_{target_class}_{len(curated):05d}.png"
                save_path = target_dir / new_name
                cv2.imwrite(str(save_path), img)
                
                curated.append({
                    "filename": new_name,
                    "class": target_class,
                    "source": "wafermap",
                    "original_class": class_dir.name
                })
                
            except Exception as e:
                continue
    
    print(f"✅ Curated {len(curated)} images from WaferMap")
    return curated


def curate_deeppcb():
    """
    Curate DeepPCB dataset
    
    Structure: DeepPCB-master/PCBData/groupXXXXX/XXXXX/ and XXXXX_not/
    - *_not folders contain NON-defective (template) images
    - Main folders contain defective images
    
    We'll extract patches from defective regions based on annotations
    """
    print("\n" + "=" * 60)
    print("🔧 Curating DeepPCB Dataset...")
    print("=" * 60)
    
    source_dir = RAW_DATA_DIR / "deeppcb" / "DeepPCB-master" / "PCBData"
    if not source_dir.exists():
        print("❌ DeepPCB not found. Skipping...")
        return []
    
    curated = []
    
    # Process each group
    for group_dir in source_dir.iterdir():
        if not group_dir.is_dir() or not group_dir.name.startswith("group"):
            continue
        
        # Find defective images folder (not *_not)
        for subfolder in group_dir.iterdir():
            if not subfolder.is_dir():
                continue
            if "_not" in subfolder.name:
                # Template images - use as "clean" samples
                target_class = "clean"
            else:
                # Defective images
                target_class = "pattern_defects"  # PCB defects map to pattern
            
            target_dir = PROCESSED_DATA_DIR / target_class
            target_dir.mkdir(parents=True, exist_ok=True)
            
            images = list(subfolder.glob("*.jpg")) + list(subfolder.glob("*.png"))
            
            for img_path in images:
                try:
                    # Skip annotation files
                    if "txt" in str(img_path) or img_path.suffix == ".txt":
                        continue
                    
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    # Extract random patches from large PCB images
                    h, w = img.shape[:2]
                    patch_size = 224
                    
                    # Extract up to 4 patches per image
                    num_patches = min(4, max(1, (h * w) // (patch_size * patch_size * 4)))
                    
                    for _ in range(num_patches):
                        if h > patch_size and w > patch_size:
                            y = np.random.randint(0, h - patch_size)
                            x = np.random.randint(0, w - patch_size)
                            patch = img[y:y+patch_size, x:x+patch_size]
                        else:
                            patch = cv2.resize(img, (patch_size, patch_size))
                        
                        patch = preprocess_image(patch)
                        
                        new_name = f"pcb_{target_class}_{len(curated):05d}.png"
                        save_path = target_dir / new_name
                        cv2.imwrite(str(save_path), patch)
                        
                        curated.append({
                            "filename": new_name,
                            "class": target_class,
                            "source": "deeppcb",
                            "original_class": "template" if "_not" in subfolder.name else "defect"
                        })
                        
                except Exception as e:
                    continue
    
    print(f"✅ Curated {len(curated)} images from DeepPCB")
    return curated


def curate_wm811k():
    """
    Curate WM811K dataset from pickle file
    
    The pickle file contains wafer maps with labels
    """
    print("\n" + "=" * 60)
    print("🔧 Curating WM811K Dataset...")
    print("=" * 60)
    
    pkl_path = RAW_DATA_DIR / "wm811k" / "LSWMD.pkl"
    if not pkl_path.exists():
        print("❌ WM811K pickle not found. Skipping...")
        return []
    
    # Class mapping for WM811K
    class_mapping = {
        0: "clean",        # None
        1: "center_defects",  # Center
        2: "center_defects",  # Donut
        3: "edge_defects",    # Edge-Loc
        4: "edge_defects",    # Edge-Ring
        5: "pattern_defects", # Loc
        6: "pattern_defects", # Near-Full
        7: "random_defects",  # Random
        8: "scratches",       # Scratch
    }
    
    # Also map by name
    name_mapping = {
        "none": "clean",
        "center": "center_defects",
        "donut": "center_defects",
        "edge-loc": "edge_defects",
        "edge-ring": "edge_defects",
        "loc": "pattern_defects",
        "near-full": "pattern_defects",
        "random": "random_defects",
        "scratch": "scratches",
    }
    
    curated = []
    
    try:
        print("  Loading pickle file (this may take a while)...")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # WM811K typically has waferMap, failureType columns
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            print("  Unexpected data format")
            return []
        
        print(f"  Found {len(df)} wafer maps")
        
        # Sample to avoid too many images (limit to 500 per class for balance)
        class_counts = defaultdict(int)
        max_per_class = 400
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="WM811K"):
            try:
                # Get wafer map
                wafer_map = row.get('waferMap', None)
                if wafer_map is None:
                    continue
                
                # Get failure type
                failure_type = row.get('failureType', None)
                if failure_type is None:
                    continue
                
                # Convert failure type to our class
                if isinstance(failure_type, (list, np.ndarray)):
                    # One-hot encoded
                    failure_idx = np.argmax(failure_type)
                    target_class = class_mapping.get(failure_idx, "other")
                elif isinstance(failure_type, str):
                    target_class = name_mapping.get(failure_type.lower().strip(), "other")
                else:
                    target_class = "other"
                
                # Check class limit
                if class_counts[target_class] >= max_per_class:
                    continue
                
                # Convert wafer map to image
                wafer_img = np.array(wafer_map).astype(np.uint8)
                
                # Normalize to 0-255
                if wafer_img.max() > 0:
                    wafer_img = (wafer_img / wafer_img.max() * 255).astype(np.uint8)
                
                # Resize to target size
                wafer_img = cv2.resize(wafer_img, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                
                # Apply CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                wafer_img = clahe.apply(wafer_img)
                
                # Save
                target_dir = PROCESSED_DATA_DIR / target_class
                target_dir.mkdir(parents=True, exist_ok=True)
                
                new_name = f"wm811k_{target_class}_{len(curated):05d}.png"
                save_path = target_dir / new_name
                cv2.imwrite(str(save_path), wafer_img)
                
                curated.append({
                    "filename": new_name,
                    "class": target_class,
                    "source": "wm811k",
                    "original_class": str(failure_type)
                })
                
                class_counts[target_class] += 1
                
            except Exception as e:
                continue
        
    except Exception as e:
        print(f"  Error loading WM811K: {e}")
        return []
    
    print(f"✅ Curated {len(curated)} images from WM811K")
    return curated


def print_class_distribution(metadata: list):
    """Print distribution of images across classes"""
    class_counts = defaultdict(int)
    source_counts = defaultdict(int)
    
    for item in metadata:
        class_counts[item['class']] += 1
        source_counts[item['source']] += 1
    
    print("\n" + "=" * 60)
    print("📊 DATASET DISTRIBUTION")
    print("=" * 60)
    
    print("\nBy Class:")
    total = 0
    for cls in CLASS_NAMES:
        count = class_counts.get(cls, 0)
        bar = "█" * (count // 20)
        print(f"  {cls:20s}: {count:4d} {bar}")
        total += count
    
    print(f"\n  {'TOTAL':20s}: {total:4d}")
    
    print("\nBy Source:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source:15s}: {count:4d}")


def main():
    """Main curation orchestration"""
    print("=" * 60)
    print("🔬 SEMICONDUCTOR DEFECT DETECTION - DATA CURATION")
    print("=" * 60)
    
    # Create directories
    for cls in CLASS_NAMES:
        (PROCESSED_DATA_DIR / cls).mkdir(parents=True, exist_ok=True)
    
    # Curate each dataset
    all_metadata = []
    
    # WaferMap (most relevant - already organized)
    metadata = curate_wafermap()
    all_metadata.extend(metadata)
    
    # DeepPCB (PCB defects - supplementary)
    metadata = curate_deeppcb()
    all_metadata.extend(metadata)
    
    # WM811K (if pickle file is processable)
    metadata = curate_wm811k()
    all_metadata.extend(metadata)
    
    # Print distribution
    print_class_distribution(all_metadata)
    
    # Save metadata CSV
    if all_metadata:
        metadata_df = pd.DataFrame(all_metadata)
        metadata_path = PROCESSED_DATA_DIR / "metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        print(f"\n💾 Metadata saved to: {metadata_path}")
        
        # Check if we have enough data
        class_counts = metadata_df['class'].value_counts()
        min_count = class_counts.min()
        max_count = class_counts.max()
        
        print(f"\n📈 Class balance: min={min_count}, max={max_count}")
        
        if min_count < 100:
            print("\n⚠️ WARNING: Some classes have fewer than 100 images!")
            print("Augmentation will help balance the dataset.")
    else:
        print("\n❌ No images were curated! Check the raw data directory.")
        return
    
    print("\n🎉 Curation complete! Next step:")
    print("   python data/augment.py")


if __name__ == "__main__":
    main()
