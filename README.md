# 🔍 SignalScope — AI-Generated Image Detection System

SignalScope is a production-ready local inference application designed to screen images and classify whether an uploaded image is likely an **AI-generated synthetic image (FAKE)** or an **authentic photograph (REAL)**. The application utilizes a ResNet-50 deep learning model fine-tuned on the **CIFAKE dataset**.

---

## 📌 Features

- **Binary Classification:** Classifies input images into `FAKE` (AI-generated) or `REAL` (Real photograph).
- **Live PyTorch Diagnostics:** Displays raw unrounded model logits, Softmax probabilities, device placement, and inference latency in real-time to verify dynamic compute.
- **Probabilistic Output:** Provides class probabilities (Fake vs. Real) and confidence percentage.
- **Confidence Interpretability:** Categorizes results into High, Moderate, or Low confidence with clear warnings for ambiguous images.
- **Streamlit Web UI:** Interactive single-image analysis and batch image upload support with CSV export.
- **Robust Preprocessing:** Handles RGB, Grayscale, and RGBA images with automatic EXIF orientation adjustment.
- **Error Handling:** Suppresses technical tracebacks and displays user-friendly error guidance.

---

## 🔬 Model & Dataset Information

- **Model Architecture:** ResNet-50 (Transfer Learning with ImageNet initial weights)
- **Dataset:** CIFAKE (Image Resolution $224 \times 224$)
- **Classes:** `FAKE` (Index 0), `REAL` (Index 1)
- **Normalization:** ImageNet Mean `[0.485, 0.456, 0.406]`, Std `[0.229, 0.224, 0.225]`
- **Optimizer:** AdamW ($1 \times 10^{-4}$ learning rate) with Cosine Annealing scheduler

---

## 📊 Benchmark Performance

The performance below was evaluated on the official **CIFAKE test dataset** (20,000 images):

| Metric | Value |
| :--- | :---: |
| **Test Accuracy** | **97.42%** |
| **Macro F1-Score** | **0.9741** |
| **ROC-AUC** | **0.9964** |
| **PR-AUC** | **0.9966** |
| **Sensitivity (Recall)** | **97.00%** |
| **Specificity** | **97.83%** |
| **Inference Speed** | **~1092 img/sec (GPU Batch Inference)** |

> [!WARNING]
> **Dataset Domain Limitation**: This model was trained strictly on the **CIFAKE dataset** ($32 \times 32$ CIFAR-10 images upscaled to $224 \times 224$). High-resolution digital camera photographs or modern webcam photos possess different spatial frequency characteristics than CIFAR-10 real photos, which can cause modern webcam photos to trigger high `FAKE` activations due to dataset domain shift.

---

## 📁 Project Structure

```
ALGONAUTS_INTERNAL_LJ_HACK-main/
│
├── model/
│   └── best_resnet50_cifake.pth       # Trained PyTorch ResNet-50 checkpoint (~94.3 MB)
│
├── evaluation/
│   ├── final_evaluation_metrics.csv   # Numerical test metrics export
│   └── test_predictions.csv           # Detailed 20k test predictions & confidence scores
│
├── outputs/
│   ├── confusion_matrix.png           # Confusion matrix chart
│   ├── precision_recall_curve.png     # PR curve plot
│   ├── roc_curve.png                  # ROC curve plot
│   ├── training_validation_accuracy.png
│   └── training_validation_loss.png
│
├── app/
│   ├── __init__.py                    # Package marker
│   ├── app.py                         # Streamlit web application interface
│   ├── config.py                      # Global paths, constants, and metric settings
│   ├── model_loader.py                # PyTorch checkpoint loader with caching
│   └── predictor.py                   # Image preprocessing & inference engine
│
├── tests/
│   ├── __init__.py                    # Test package marker
│   └── test_predictor.py              # Pytest unit tests
│
├── notebook/
│   └── SignalScope_CIFAKE_ResNet50_Native32.ipynb # Training & evaluation notebook
│
├── .venv/                             # Local Python virtual environment
├── requirements.txt                   # Dependency list
├── README.md                          # Project documentation
├── rules.md                           # Team collaboration & project guidelines
└── .gitignore                         # Git exclusion rules
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup

Open PowerShell and navigate to the project directory:

```powershell
# Create virtual environment (if not already created)
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Run the Streamlit Application

Launch the local web application:

```powershell
streamlit run app/app.py
```

The app will launch in your web browser automatically at `http://localhost:8501`.

### 3. Run Unit Tests

Verify the model loader and inference engine with Pytest:

```powershell
pytest tests/test_predictor.py -v
```

---

## 🛡️ User & Safety Disclaimer

*SignalScope is designed as an AI screening and assistance tool. Prediction scores reflect model probabilities based on visual artifact distributions in the training dataset and should not be used as sole legal or definitive proof of image authenticity.*
