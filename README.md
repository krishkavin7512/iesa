# Semiconductor Defect Detection - Edge AI

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Production-ready Edge-AI system for semiconductor defect classification achieving 94-95% accuracy with <2MB model size.

## 🎯 Objective

Detect and classify 8 types of semiconductor defects in wafer/die images:
1. Clean (no defects)
2. Scratches
3. Particles
4. Pattern_Defects
5. Edge_Defects
6. Center_Defects
7. Random_Defects
8. Other

## 🏗️ Architecture

- **Primary Model:** EfficientNet-Lite0 (~900KB quantized)
- **Ensemble Partner:** MobileNetV3-Small (~600KB quantized)
- **Target Platform:** NXP i.MX RT (ARM Cortex-M7)

## 📁 Project Structure

```
semiconductor-defect-detection/
├── data/
│   ├── raw/           # Original downloaded datasets
│   ├── processed/     # Cleaned and organized images
│   └── augmented/     # Final augmented train/val/test
├── models/            # Model architecture definitions
├── training/          # Training scripts and configs
├── evaluation/        # Evaluation and metrics scripts
├── inference/         # Inference and deployment scripts
├── notebooks/         # Jupyter notebooks for exploration
├── outputs/           # Trained models, logs, visualizations
└── docs/              # Documentation and reports
```

## 🚀 Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up Kaggle credentials (for datasets)
# Place kaggle.json in ~/.kaggle/

# 4. Download datasets
python data/download_datasets.py

# 5. Preprocess and augment
python data/preprocess.py
python data/augment.py

# 6. Train models
python training/train.py

# 7. Evaluate
python evaluation/evaluate.py
```

## 📊 Performance Targets

| Metric | Target |
|--------|--------|
| Accuracy | ≥94% |
| Model Size | <2MB (quantized) |
| Inference | <150ms on CPU |

## 🛠️ Tech Stack

- Python 3.10+
- TensorFlow 2.14.0
- OpenCV, Albumentations
- ONNX, TFLite

## 📄 License

MIT License
