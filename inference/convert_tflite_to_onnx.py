import tflite2onnx
import os

tflite_path = 'submission/model/model.tflite'
onnx_path = 'submission/model/model.onnx'

if not os.path.exists(tflite_path):
    print(f"Error: {tflite_path} not found.")
    exit(1)

print(f"Converting {tflite_path} to {onnx_path}...")
tflite2onnx.convert(tflite_path, onnx_path)
print("Conversion successful!")
