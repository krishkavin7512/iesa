"""
Dataset Download Script for Semiconductor Defect Detection

This script downloads and organizes datasets from multiple sources:
1. MixedWM38 (Kaggle) - Wafer Map defects
2. DeepPCB (GitHub) - PCB defects
3. Severstal (Kaggle) - Steel surface defects

Run: python data/download_datasets.py
"""

import os
import sys
import zipfile
import shutil
import subprocess
from pathlib import Path
import urllib.request
from tqdm import tqdm

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: str):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def check_kaggle_setup():
    """Check if Kaggle API is configured"""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("=" * 60)
        print("❌ Kaggle API not configured!")
        print("=" * 60)
        print("\nPlease follow these steps:")
        print("1. Go to https://www.kaggle.com/")
        print("2. Create an account (or log in)")
        print("3. Go to Settings → API → Create New Token")
        print("4. Download kaggle.json")
        print(f"5. Place it in: {kaggle_json.parent}")
        print("\nSee docs/KAGGLE_SETUP.md for detailed instructions.")
        print("=" * 60)
        return False
    return True


def download_mixedwm38():
    """Download MixedWM38 wafer map dataset from Kaggle"""
    print("\n" + "=" * 60)
    print("📥 Downloading MixedWM38 Wafer Map Dataset...")
    print("=" * 60)
    
    output_dir = RAW_DATA_DIR / "mixedwm38"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Use Kaggle API
        cmd = f'kaggle datasets download -d qingyi/mixedwm38 -p "{output_dir}" --unzip'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ MixedWM38 downloaded successfully!")
            return True
        else:
            print(f"❌ Download failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def download_deeppcb():
    """Download DeepPCB dataset from GitHub"""
    print("\n" + "=" * 60)
    print("📥 Downloading DeepPCB Dataset...")
    print("=" * 60)
    
    output_dir = RAW_DATA_DIR / "deeppcb"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # DeepPCB is on GitHub - using the main dataset
    url = "https://github.com/tangsanli5201/DeepPCB/archive/refs/heads/master.zip"
    zip_path = output_dir / "deeppcb.zip"
    
    try:
        print(f"Downloading from GitHub...")
        download_url(url, str(zip_path))
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        # Clean up zip
        zip_path.unlink()
        print("✅ DeepPCB downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def download_severstal():
    """Download Severstal Steel Defect dataset from Kaggle"""
    print("\n" + "=" * 60)
    print("📥 Downloading Severstal Steel Defect Dataset...")
    print("=" * 60)
    
    output_dir = RAW_DATA_DIR / "severstal"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Need to accept competition rules first
        cmd = f'kaggle competitions download -c severstal-steel-defect-detection -p "{output_dir}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Unzip if needed
            for zip_file in output_dir.glob("*.zip"):
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
                zip_file.unlink()
            print("✅ Severstal downloaded successfully!")
            return True
        else:
            print(f"⚠️ Severstal download issue: {result.stderr}")
            print("Note: You may need to accept competition rules at:")
            print("https://www.kaggle.com/c/severstal-steel-defect-detection/rules")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def create_class_directories():
    """Create the 8 class directories for processed data"""
    classes = [
        "clean",
        "scratches", 
        "particles",
        "pattern_defects",
        "edge_defects",
        "center_defects",
        "random_defects",
        "other"
    ]
    
    for split in ["train", "val", "test"]:
        for cls in classes:
            dir_path = PROJECT_ROOT / "data" / "augmented" / split / cls
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # Also for processed (pre-augmentation)
    for cls in classes:
        dir_path = PROCESSED_DATA_DIR / cls
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("✅ Class directories created!")


def main():
    """Main download orchestration"""
    print("=" * 60)
    print("🔬 SEMICONDUCTOR DEFECT DETECTION - DATASET DOWNLOADER")
    print("=" * 60)
    
    # Create directories
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check Kaggle setup
    kaggle_ok = check_kaggle_setup()
    
    # Create class directories
    create_class_directories()
    
    # Download datasets
    results = {}
    
    # DeepPCB (no Kaggle needed)
    results['deeppcb'] = download_deeppcb()
    
    if kaggle_ok:
        # MixedWM38
        results['mixedwm38'] = download_mixedwm38()
        
        # Severstal
        results['severstal'] = download_severstal()
    else:
        print("\n⚠️ Skipping Kaggle datasets. Please set up Kaggle first!")
        results['mixedwm38'] = False
        results['severstal'] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 60)
    for dataset, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {dataset}")
    
    print("\n📍 Raw datasets location:", RAW_DATA_DIR)
    print("📍 Processed data location:", PROCESSED_DATA_DIR)
    
    if not all(results.values()):
        print("\n⚠️ Some downloads failed. Check the errors above.")
        print("You can re-run this script after fixing issues.")
    else:
        print("\n🎉 All datasets downloaded! Run preprocessing next:")
        print("   python data/preprocess.py")


if __name__ == "__main__":
    main()
