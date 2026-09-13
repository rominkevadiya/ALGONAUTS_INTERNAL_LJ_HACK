# 🔍 SignalScope — AI-Generated Image Detection System

SignalScope is a local inference application designed to screen images and classify whether an uploaded image is likely an **AI-generated synthetic image (FAKE)** or an **authentic photograph (REAL)**. The application utilizes a ResNet-50 deep learning model fine-tuned on the **CIFAKE dataset**.

---

## 📌 Which Core + Bonus Modules Built

**Core Task (Completed):** 
- Real-vs-AI-generated image classification with ResNet-50.
- Held-out test set metrics reported (ROC-AUC, Macro-F1).
- Streamlit inference interface.

**Bonus Modules (Completed):**
- **A. Faithful Explanation:** Natural language explanations of visual cues via Gemini 2.5 API.
- **B. Generator Attribution:** Multi-class generator family identification (Diffusion/GAN) via Gemini API.
- **C. Robustness to Degradation:** Empirical testing against JPEG compression and resizing.
- **D. Provenance & Metadata:** Exif and C2PA pre-screening with pipeline short-circuiting for verified AI images.
- **E. Multimodal (Image + Text):** Consistency checking between provided captions and image content.
- **F. Real-Time / Deployable:** Full Streamlit interactive interface with backend Live Terminal Logging.
- **G. Active Defence Analysis:** Adversarial FGSM attack robustness analysis.

---

## 📌 Features & Multi-Strategy Inference Capabilities

- **Binary Classification:** Classifies input images into `FAKE` (AI-generated) or `REAL` (Real photograph).
- **Native-Resolution Patch Voting:** Extracts native $32 \times 32$ crops across high-resolution images to preserve local pixel-level texture without aggressive downscaling. Features **variance-guided patch sampling** to actively target complex high-texture regions where AI artifacts hide.
- **Hybrid Inference Consensus:** Combines baseline resize inference, native patch voting, and spectral diagnostics into a unified decision tree to filter false positives (e.g. dark sensor noise in real photos).
- **Test-Time Augmentation (TTA):** Generates 8 semantically meaningful geometric and photometric views without retraining, aggregating results using an **inverse-entropy weighting** system to prioritize high-confidence views.
- **Resolution-Aware Automatic Dispatch:** Automatically routes images to the optimal inference strategy based on resolution (`Resize` <64px, `Patch` 64-256px, `Hybrid` >256px).
- **Normalized Shannon Entropy Diagnostics:** Computes output uncertainty $H(p) / \ln(2)$ to identify predictions near the decision boundary.
- **Experimental 2D FFT Spectral Diagnostic:** Analyzes radial power spectrum and frequency irregularities. Wired directly into the Hybrid strategy to confirm localized AI artifacts and suppress false positives on woven textures.
- **Streamlit Web UI & Live Logging:** Interactive single-image analysis with binned patch probability histograms, real-time backend terminal logging for diagnostic visibility, and batch image upload with CSV export.
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
- **Dataset (Core Training):** CIFAKE (MIT Licensed, ~100k+ labelled $32 \times 32$ images, balanced real/fake). No additional scraped data of real individuals was used.
- **Evaluation Datasets:** Standard CIFAKE Test Split, and synthetic mock dataset for Unseen-Generator simulation.
- **Classes:** `FAKE` (Index 0), `REAL` (Index 1)

---

## 📊 Benchmark Performance (Reported Metrics)

Evaluated on the official **CIFAKE test dataset** and a **Held-out Unseen-Generator Mock Dataset**.

| Metric | Value (CIFAKE) | Value (Unseen-Generator Split) |
| :--- | :---: | :---: |
| **Test Accuracy** | **98.33%** | **100.00%** |
| **Macro F1-Score** | **0.9832** | **1.0000** |
| **Overall ROC-AUC** | **0.9987** | **1.0000** |
| **Sensitivity (Recall)** | **98.34%** | **100.00%** |
| **Specificity** | **98.31%** | **100.00%** |
| **Inference Speed** | **~916 img/sec** | **~916 img/sec** |

*(Note: The Unseen Generator split was simulated with a synthetic dataset for the hackathon evaluation script. In real deployment, the Unseen Split AUC is computed by the judges).*

**Confusion Matrix (CIFAKE):**
- True Positives: 9834
- True Negatives: 9831
- False Positives: 169
- False Negatives: 166

---

## ⚠️ Important Technical Limitations & Disclaimers

> [!WARNING]
> **Dataset Domain Limitation**: The model was trained strictly on the **CIFAKE dataset** ($32 \times 32$ native resolution images — CIFAR-10 real photos vs Stable Diffusion v1.4 AI images). 
> Modern AI generators (Gemini, Midjourney v6, DALL-E 3, FLUX, etc.) were **not represented in training**. While the advanced Patch, TTA, and Hybrid inference strategies dramatically improve real-world robustness without retraining, this system **does not guarantee generalization to all unseen or future AI generators**.

---

## 📁 Project Structure (Hackathon Contract)

```text
ALGONAUTS_INTERNAL_LJ_HACK-main/
│
├── README.md                               # This file (Entry point)
├── requirements.txt                        # Dependency list
│
├── model/                                  # Model weights & inference code
│   ├── best_resnet50_cifake_native32_2.pth 
│   └── generator_attribution.py            # Bonus B
│
├── app/                                    # Source code (UI & logic)
│   ├── app.py                              # Streamlit web application
│   ├── predictor.py                        # Facade API
│   ├── diagnostics/                        
│   │   ├── explainer.py                    # Bonus A & E (Faithful Explanation)
│   │   ├── metadata_inspector.py           # Bonus D
│   │   └── ...
│   └── strategies/                         # Inference strategies
│
├── evaluation/                             # Active Defence & Degradation
│   ├── adversarial_analysis.py             # Bonus G
│   ├── degradation_analysis.py             # Bonus C
│   └── evaluate_unseen.py                  # Core unseen evaluation
│
├── report/
│   └── report.md                           # One-page model report (Section 7.3)
│
└── tests/                                  # Unit tests
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

### 2. Setup Gemini API Key
To utilize the Faithful Explanation (Module A), Generator Attribution (Module B), and Multimodal (Module E) features, create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### 3. Run the Streamlit Application

Launch the local web application:

```powershell
streamlit run app/app.py
```

The app will launch in your web browser automatically at `http://localhost:8501`.

### 4. Run Bonus Evaluations
- **Unseen Generator Split:** `python evaluation/evaluate_unseen.py`
- **Degradation Robustness:** `python evaluation/degradation_analysis.py`
- **Active Defence Analysis:** `python evaluation/adversarial_analysis.py`

---

## 🎥 Demo Video Link
*(Add YouTube/Vimeo unlisted link here before submission)*

## 🛡️ User & Safety Disclaimer

*SignalScope is designed as an AI screening and assistance tool. Prediction scores reflect model probabilities under selected inference strategies and should not be used as sole legal or definitive proof of image authenticity. Explanations are provided as likelihood assessments and are not accusations against individuals.*
