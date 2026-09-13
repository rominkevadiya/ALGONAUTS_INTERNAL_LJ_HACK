# 🔍 SignalScope — AI-Generated Image Detection System

SignalScope is a local inference application designed to screen images and classify whether an uploaded image is likely an **AI-generated synthetic image (FAKE)** or an **authentic photograph (REAL)**. The application utilizes a ResNet-50 deep learning model fine-tuned on the **CIFAKE dataset**.

---

## 📌 Features & Multi-Strategy Inference Capabilities

- **Binary Classification:** Classifies input images into `FAKE` (AI-generated) or `REAL` (Real photograph).
- **Native-Resolution Patch Voting:** Extracts native $32 \times 32$ crops across high-resolution images to preserve local pixel-level texture without aggressive downscaling.
- **Hybrid Inference Strategy:** Combines baseline resize inference and native patch voting to evaluate strategy consistency.
- **Test-Time Augmentation (TTA):** Evaluates predictions across multiple geometric transformations without retraining.
- **Resolution-Aware Automatic Selection:** Automatically dispatches the optimal inference mode (`Resize` for small thumbnails, `Patch` for high-resolution images).
- **Normalized Shannon Entropy Diagnostics:** Computes output uncertainty $H(p) / \ln(2)$ to identify predictions near the decision boundary.
- **Statistical Disagreement Analysis:** Measures mean, median, standard deviation, and patch agreement percentage.
- **Experimental 2D FFT Spectral Diagnostic:** Analyzes radial power spectrum and high-to-low frequency ratios as a secondary signal.
- **Streamlit Web UI:** Interactive single-image analysis with binned patch probability histograms and batch image upload with CSV export.
- **Robust Preprocessing:** Handles RGB, Grayscale, RGBA, LA, and Palette images with alpha background compositing and EXIF rotation handling.

---

## ⚙️ Inference Modes & Usage Guide

SignalScope supports five inference modes without retraining or mutating the underlying model checkpoint:

1. **Automatic (`auto`):** Resolution-aware mode. Automatically uses `Resize` mode for small images ($<128\text{px}$) and `Patch` mode for high-resolution images ($\ge 128\text{px}$).
2. **Resize (`resize`):** Baseline pipeline. Resizes the full image directly to $32 \times 32$ using bicubic interpolation.
3. **Native Patch Voting (`patch`):** Extracts $N$ native $32 \times 32$ crops (center, corners, grid, and random) at native pixel resolution. Classifies patches in a single batched pass and aggregates probabilities (`mean`, `median`, `majority`, `logit_mean`).
4. **Hybrid (`hybrid`):** Runs both `Resize` and `Patch` inference, reporting probability difference and agreement status (*Strong*, *Partial*, *Disagreement*).
5. **Test-Time Augmentation (`tta`):** Runs inference across multiple geometric views and reports prediction standard deviation.

---

## 🔬 Model & Dataset Information

- **Model Architecture:** ResNet-50 (Adapted Native 32x32 Stem)
- **Dataset:** CIFAKE (Image Resolution $32 \times 32$)
- **Classes:** `FAKE` (Index 0), `REAL` (Index 1)
- **Normalization:** ImageNet Mean `[0.485, 0.456, 0.406]`, Std `[0.229, 0.224, 0.225]`

---

## 📊 Benchmark Performance

The performance below was evaluated on the official **CIFAKE test dataset** (20,000 images):

| Metric | Value |
| :--- | :---: |
| **Test Accuracy** | **98.33%** |
| **Macro F1-Score** | **0.9832** |
| **ROC-AUC** | **0.9987** |
| **PR-AUC** | **0.9988** |
| **Sensitivity (Recall)** | **98.34%** |
| **Specificity** | **98.31%** |
| **Inference Speed** | **~916 img/sec (GPU Batch Inference)** |

---

## ⚠️ Important Technical Limitations & Disclaimers

> [!WARNING]
> **Dataset Domain Limitation**: The model was trained strictly on the **CIFAKE dataset** ($32 \times 32$ native resolution images — CIFAR-10 real photos vs Stable Diffusion v1.4 AI images). 
> Modern AI generators (Gemini, Midjourney v6, DALL-E 3, FLUX, etc.) were **not represented in training**. While native patch inference and multi-strategy evaluation improve resolution robustness, this system **does not guarantee generalization to all unseen or future AI generators**.

---

## 📁 Project Structure

```
ALGONAUTS_INTERNAL_LJ_HACK-main/
│
├── model/
│   └── best_resnet50_cifake_native32_2.pth # Trained PyTorch ResNet-50 checkpoint (~94.3 MB)
│
├── evaluation/
│   ├── final_evaluation_metrics.csv        # Numerical test metrics export
│   ├── inference_strategy_benchmark.csv    # Strategy benchmark comparison
│   ├── run_experimental_evaluation.py     # Non-retraining benchmark script
│   └── test_predictions.csv                # Detailed 20k test predictions
│
├── app/
│   ├── __init__.py                         # Package marker
│   ├── app.py                              # Streamlit web application interface
│   ├── config.py                           # Global paths, constants, and metric settings
│   ├── model_loader.py                     # PyTorch checkpoint loader with caching
│   └── predictor.py                        # Multi-strategy inference engine & diagnostics
│
├── tests/
│   ├── __init__.py                         # Test package marker
│   ├── test_patch_inference.py             # Complete unit tests (Patch, TTA, Hybrid, FFT, Entropy)
│   └── test_predictor.py                   # Pytest baseline unit tests
│
├── .venv/                                  # Local Python virtual environment
├── requirements.txt                        # Dependency list
├── README.md                               # Project documentation
└── rules.md                                # Team collaboration & project guidelines
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup

Open PowerShell and navigate to the project directory:

```powershell
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

Verify the model loader and multi-strategy inference engine with Pytest:

```powershell
pytest tests/ -v
```

---

## 🛡️ User & Safety Disclaimer

*SignalScope is designed as an AI screening and assistance tool. Prediction scores reflect model probabilities under selected inference strategies and should not be used as sole legal or definitive proof of image authenticity.*

