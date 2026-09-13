# 🔍 SignalScope — AI-Generated Image Detection System

SignalScope is a local inference application designed to screen images and classify whether an uploaded image is likely an **AI-generated synthetic image (FAKE)** or an **authentic photograph (REAL)**. The application utilizes a ResNet-50 deep learning model fine-tuned on the **CIFAKE dataset**.

---

## 📌 Features & Multi-Strategy Inference Capabilities

- **Binary Classification:** Classifies input images into `FAKE` (AI-generated) or `REAL` (Real photograph).
- **Native-Resolution Patch Voting:** Extracts native $32 \times 32$ crops across high-resolution images to preserve local pixel-level texture without aggressive downscaling. Features **variance-guided patch sampling** to actively target complex high-texture regions where AI artifacts hide.
- **Hybrid Inference Consensus:** Combines baseline resize inference, native patch voting, and spectral diagnostics into a unified decision tree to filter false positives (e.g. dark sensor noise in real photos).
- **Test-Time Augmentation (TTA):** Generates 8 semantically meaningful geometric and photometric views without retraining, aggregating results using an **inverse-entropy weighting** system to prioritize high-confidence views.
- **Resolution-Aware Automatic Dispatch:** Automatically routes images to the optimal inference strategy based on resolution (`Resize` <64px, `Patch` 64-256px, `Hybrid` >256px).
- **Normalized Shannon Entropy Diagnostics:** Computes output uncertainty $H(p) / \ln(2)$ to identify predictions near the decision boundary.
- **Experimental 2D FFT Spectral Diagnostic:** Analyzes radial power spectrum and frequency irregularities. Wired directly into the Hybrid strategy to confirm localized AI artifacts.
- **Streamlit Web UI:** Interactive single-image analysis with binned patch probability histograms and batch image upload with CSV export.

---

## ⚙️ Inference Modes & Usage Guide

SignalScope utilizes a modular facade architecture (`app.predictor`) that delegates to five specialized inference modes without mutating the underlying model checkpoint:

1. **Automatic (`auto`):** Resolution-aware graduated dispatcher.
   - `<64px`: Uses Baseline Resize
   - `64px - 256px`: Uses Native Patch Vote
   - `>256px`: Uses Hybrid Consensus (Full Analysis)
2. **Resize (`resize`):** Baseline pipeline. Resizes the full image directly to $32 \times 32$ using bicubic interpolation. Best for very small CIFAKE-native images.
3. **Native Patch Voting (`patch`):** Extracts $N$ native $32 \times 32$ crops at native pixel resolution. Uses **adaptive luminance thresholds** to filter dark noise and applies **center-weighting aggregation** (salient zone focus) for final probability.
4. **Hybrid (`hybrid`):** Runs both `Resize` and `Patch` inference, cross-referenced with `FFT Spectral` scores. Employs a complex decision tree to filter aliasing false-positives and confirm localized AI artifacts.
5. **Test-Time Augmentation (`tta`):** Runs inference across 8 augmented views (Original, H-Flip, Center Crop, Brightness ±15%, Contrast +20%, Rotate 90°/45°). Evaluates prediction stability and self-consistency.

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
> Modern AI generators (Gemini, Midjourney v6, DALL-E 3, FLUX, etc.) were **not represented in training**. While the advanced Patch, TTA, and Hybrid inference strategies dramatically improve real-world robustness without retraining, this system **does not guarantee generalization to all unseen or future AI generators**.

---

## 📁 Project Structure

```text
ALGONAUTS_INTERNAL_LJ_HACK-main/
│
├── model/
│   └── best_resnet50_cifake_native32_2.pth # Trained PyTorch ResNet-50 checkpoint (~94.3 MB)
│
├── evaluation/
│   ├── run_experimental_evaluation.py      # Non-retraining benchmark script
│   └── ...                                 # Historical CSV metrics
│
├── app/
│   ├── app.py                              # Streamlit web application interface
│   ├── config.py                           # Global paths, constants, and thresholds
│   ├── model_loader.py                     # PyTorch checkpoint loader with caching
│   ├── predictor.py                        # Facade API for inference strategies
│   ├── diagnostics/                        # Analytics modules
│   │   ├── entropy.py                      # Shannon entropy calculations
│   │   ├── disagreement.py                 # Statistical variance and stability
│   │   └── fft_spectral.py                 # 2D Fast Fourier Transform scoring
│   └── strategies/                         # Inference strategy modules
│       ├── auto/                           # Graduated resolution dispatcher
│       ├── hybrid/                         # Multi-strategy consensus decision tree
│       ├── patch/                          # Variance-guided native patch extraction
│       ├── resize/                         # Baseline 32x32 scaling
│       └── tta/                            # 8-view test-time augmentation
│
├── tests/
│   ├── test_patch_inference.py             # 20+ Unit tests for all inference modes
│   └── test_predictor.py                   # Baseline structural tests
│
├── .venv/                                  # Local Python virtual environment
├── requirements.txt                        # Dependency list
└── README.md                               # Project documentation
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

Verify the model loader and multi-strategy inference engine with Pytest (24 tests):

```powershell
pytest tests/ -v
```

---

## 🛡️ User & Safety Disclaimer

*SignalScope is designed as an AI screening and assistance tool. Prediction scores reflect model probabilities under selected inference strategies and should not be used as sole legal or definitive proof of image authenticity.*
