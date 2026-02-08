"""
Script to verify model loading in TF 2.10 environment
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, MobileNetV3Small

print(f"TensorFlow Version: {tf.__version__}")

def check_efficientnet():
    print("\nChecking EfficientNetB0...")
    try:
        # TF 2.10 API check
        base = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3)
            # Removed pooling=None to check defaults
        )
        print("✅ EfficientNetB0 loaded successfully")
        return True
    except Exception as e:
        print(f"❌ EfficientNetB0 failed: {e}")
        return False

def check_mobilenet():
    print("\nChecking MobileNetV3Small...")
    try:
        # TF 2.10 API check
        base = MobileNetV3Small(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3),
            include_preprocessing=True # This might fail in 2.10
        )
        print("✅ MobileNetV3Small loaded successfully")
        return True
    except TypeError as e:
        print(f"⚠️ MobileNetV3Small type error (likely 'include_preprocessing'): {e}")
        # Try without include_preprocessing
        try:
            base = MobileNetV3Small(
                include_top=False,
                weights='imagenet',
                input_shape=(224, 224, 3)
            )
            print("✅ MobileNetV3Small loaded structure (without preprocessing arg)")
            return True
        except Exception as e2:
             print(f"❌ MobileNetV3Small completely failed: {e2}")
             return False
    except Exception as e:
        print(f"❌ MobileNetV3Small failed: {e}")
        return False

if __name__ == "__main__":
    e_ok = check_efficientnet()
    m_ok = check_mobilenet()
    
    if e_ok and m_ok:
        print("\n🎉 All models compatible!")
        sys.exit(0)
    else:
        print("\n⚠️ Some models failed verification")
        sys.exit(1)
