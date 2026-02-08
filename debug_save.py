"""
Debug script to test model saving functionality
"""
import sys
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set TF log level
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from models.efficientnet_model import create_efficientnet_model

def test_save():
    print("Creating model...")
    try:
        model = create_efficientnet_model()
        model.build((None, 224, 224, 3))
        print("Model created.")
    except Exception as e:
        print(f"FAILED to create model: {e}")
        return

    print("\n1. Attempting to save .h5 (full model)...")
    try:
        model.save("debug_model.h5")
        print("✅ Saved .h5 successfully")
    except Exception as e:
        print(f"❌ Failed to save .h5: {e}")

    print("\n2. Attempting to save .keras (full model)...")
    try:
        model.save("debug_model.keras")
        print("✅ Saved .keras successfully")
    except Exception as e:
        print(f"❌ Failed to save .keras: {e}")

    print("\n3. Attempting to save weights only (.h5)...")
    try:
        model.save_weights("debug_weights.h5")
        print("✅ Saved weights .h5 successfully")
    except Exception as e:
        print(f"❌ Failed to save weights: {e}")

if __name__ == "__main__":
    test_save()
