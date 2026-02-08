import os
import tensorflow as tf
# import tf2onnx # Skipped due to proto conflict
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.efficientnet_model import create_efficientnet_model
from training.config import INPUT_SHAPE, NUM_CLASSES

def export_models(weights_path, output_dir):
    """Export trained model to ONNX and TFLite formats"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading weights from: {weights_path}")
    
    # 1. Rebuild Model
    model = create_efficientnet_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    model.build((None, *INPUT_SHAPE))
    model.load_weights(weights_path)
    print("Model loaded successfully.")
    
    # 2. Save TFLite
    print("\n📦 Converting to TFLite (Float32)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    tflite_path = output_dir / "model.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"   Saved: {tflite_path}")
    
    # 3. Save TFLite (INT8 Quantized) - Optional/Bonus for eIQ
    # Requires representative dataset, skipping for now to keep simple script
    # But could do dynamic range quantization easily
    print("\n📦 Converting to TFLite (Dynamic Range Quantization)...")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    
    tflite_quant_path = output_dir / "model_quant.tflite"
    with open(tflite_quant_path, "wb") as f:
        f.write(tflite_quant_model)
    print(f"   Saved: {tflite_quant_path}")
    
    # 4. Save ONNX (Skipped due to env issues)
    print("\n📦 Converting to ONNX... (SKIPPED)")
    print("   -> Please use the provided Colab notebook to convert 'saved_model' to ONNX.")
    
    # 4b. Save TensorFlow SavedModel (For external conversion)
    saved_model_path = output_dir / "saved_model"
    model.save(saved_model_path)
    print(f"   Saved: {saved_model_path} (Use this for ONNX conversion)")
    
    # 5. Generate Instruction Stack (Summary)
    print("\n📄 Generating Instruction Stack...")
    summary_path = output_dir / "instruction_stack.txt"
    with open(summary_path, "w") as f:
        # Standard summary
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write("\n\n=== TFLite Operator Details ===\n")
        # TODO: Use tflite analyzer if possible, but basic summary is good start
    print(f"   Saved: {summary_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="outputs/models/efficientnet_final_weights.h5")
    parser.add_argument("--output", default="submission/model")
    args = parser.parse_args()
    
    export_models(args.weights, args.output)
