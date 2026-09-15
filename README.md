# 🔍 SignalScope — AI-Generated Image Detection System

SignalScope is a local inference application designed to screen images and classify whether an uploaded image is likely an **AI-generated synthetic image (FAKE)** or an **authentic photograph (REAL)**. The application utilizes a ResNet-50 deep learning model fine-tuned on the **CIFAKE dataset** with a CIFAR-adapted 32×32 native stem.

---

## 📑 Table of Contents
- [Which Core + Bonus Modules Built](#-which-core--bonus-modules-built)
- [Features & Multi-Strategy Capabilities](#-features--multi-strategy-inference-capabilities)
- [Inference Modes & Usage Guide](#️-inference-modes--usage-guide)
- [Model & Dataset Information](#-model--dataset-information)
- [Benchmark Performance](#-benchmark-performance-verified-metrics)
- [Technical Limitations](#️-important-technical-limitations--disclaimers)
- [Project Structure](#-project-structure)
- [Quick Start Guide](#-quick-start-guide)
- [Demo Video](#-demo-video)

---

## 📌 Which Core + Bonus Modules Built

**Core Task (Completed):**
- Real-vs-AI-generated image classification with ResNet-50.
- Held-out test set metrics reported (ROC-AUC, Macro-F1).
- Streamlit inference interface.

**Bonus Modules (Completed):**
- **A. Faithful Explanation:** Natural language explanations of visual cues via Gemini integration.
- **B. Generator Attribution:** Multi-class attribution (e.g., Adobe Firefly, Midjourney, Flux.1, Sora, Kling, Runway, Pika, SDXL, DALL-E 3) implemented via **`c2pa-python`** to decrypt and extract cryptographic Content Credentials (`softwareAgent`), falling back to EXIF signatures and Gemini multimodal analysis if absent.
- **C. Robustness to Degradation:** High resilience against JPEG compression and resizing through a **native 32×32 patch consensus** inference pipeline instead of global image downsampling.
- **D. Provenance & Metadata:** Extensive pre-screening for C2PA byte signatures (`jumbc2pa`), EXIF AI-tool tags (44 known signatures), and camera hardware tags.
- **E. Multimodal (Image + Text):** Rigorous image-caption consistency checking and JSON scoring via the Gemini API.
- **F. Real-Time / Deployable:** Full Streamlit interactive interface presenting responsible "Provenance Verdicts" and live backend terminal logging.
- **G. Active Defence Analysis:** Adversarial FGSM attack robustness analysis documented in `AUDIT_REPORT.md`.

---

## 📌 Features & Multi-Strategy Inference Capabilities

- **Binary Classification:** Classifies input images into `FAKE` (AI-generated) or `REAL` (Real photograph).
- **Native-Resolution Patch Voting:** Extracts native 32×32 crops across high-resolution images to preserve local pixel-level texture without aggressive downscaling. Features **variance-guided patch sampling** (targets high-texture regions where AI artifacts concentrate) and **center-weighted aggregation**.
- **Hybrid Inference Consensus:** Combines baseline resize inference, native patch voting, and spectral diagnostics into a calibrated decision tree to filter false positives (e.g. dark sensor noise in real photos).
- **MultiScale Strategy (3-Branch):** Runs simultaneous Global (full-image resize), Context (128×128 grid patches), and Native Texture (32×32 crops) branches with adaptive weight fusion. Best for high-resolution AI images.
- **Test-Time Augmentation (TTA):** Generates 8 semantically meaningful geometric and photometric views, aggregating results using **inverse-entropy weighting** to prioritize high-confidence views.
- **Resolution-Aware Automatic Dispatch:** Automatically routes images to the optimal inference strategy based on minimum image dimension:
  - `< 64px` → Resize (tiny image; patches impossible)
  - `64–255px` → Patch Voting (small image)
  - `256–511px` → Hybrid Consensus (medium image)
  - `≥ 512px` → MultiScale (large/high-res; best for modern AI output)
- **Normalized Shannon Entropy Diagnostics:** Computes output uncertainty H(p)/ln(2) to identify predictions near the decision boundary.
- **Experimental 2D FFT Spectral Diagnostic:** Analyzes radial power spectrum and frequency irregularities. Wired into the Hybrid strategy to confirm localized AI artifacts.


---

## ⚙️ Inference Modes & Usage Guide

SignalScope uses a modular facade architecture (`app.predictor`) delegating to six specialized inference modes:

1. **Automatic (`auto`):** Resolution-aware graduated dispatcher.
   - `< 64px` → Baseline Resize
   - `64px – 255px` → Native Patch Voting
   - `256px – 511px` → Hybrid Consensus
   - `≥ 512px` → MultiScale (3-Branch)
2. **Resize (`resize`):** Baseline pipeline. Resizes the full image to 32×32 using bicubic interpolation.
3. **Native Patch Voting (`patch`):** Extracts N native 32×32 crops. Uses **adaptive luminance thresholds** to filter dark noise and applies **center-weighting aggregation**. Top-K ratio (20%) consistent across all strategies.
4. **Hybrid (`hybrid`):** Runs both Resize and Patch inference, cross-referenced with FFT Spectral scores. Employs a calibrated decision tree to filter aliasing false-positives.
5. **MultiScale (`multiscale`):** Three-branch spatial analysis (Global + Context + Native Texture) with adaptive weight fusion and tri-state output (REAL / FAKE / UNCERTAIN).
6. **Test-Time Augmentation (`tta`):** 8-view augmentation (Original, H-Flip, Center Crop, Brightness ±15%, Contrast +20%, Rotate 90°/45°). Evaluates prediction stability via inverse-entropy weighting.

---

## 🔬 Model & Dataset Information

- **Model Architecture:** ResNet-50 (CIFAR-Adapted Native 32×32 Stem — `Conv2d(3,64,3,1,1)` + `nn.Identity()` maxpool)
- **Dataset (Core Training):** CIFAKE (MIT Licensed, 120,000 labelled 32×32 images, balanced real/fake). No additional scraped data of real individuals was used.
- **Classes:** `FAKE` (Index 0), `REAL` (Index 1)
- **Checkpoint:** `model/best_resnet50_cifake_retrained.pth` (23.57M parameters, ~90MB)

---

## 📊 Benchmark Performance (Verified Metrics)

Evaluated on the official **CIFAKE test dataset** (20,000 images). Metrics independently verified from `evaluation/test_predictions.csv`.

| Metric | Verified Value |
| :--- | :---: |
| **Test Accuracy** | **97.87%** |
| **Macro F1-Score** | **0.9786** |
| **Overall ROC-AUC** | **0.9980** |
| **Unseen-Generator Split ROC-AUC** | **1.0000** |
| **PR-AUC** | **0.9981** |
| **Sensitivity (Recall REAL)** | **97.87%** |
| **Specificity (Recall FAKE)** | **97.86%** |
| **MCC** | **0.9573** |

**Confusion Matrix (CIFAKE, 20,000 images):**
- True Positives (REAL correctly identified): **9,787**
- True Negatives (FAKE correctly identified): **9,786**
- False Positives (FAKE labelled REAL): **214**
- False Negatives (REAL labelled FAKE): **213**

> [!NOTE]
> Metrics above are benchmarks on the CIFAKE test set. Real-world performance on modern high-resolution AI generators (Midjourney v6, Flux.1, DALL-E 3) may differ due to dataset domain shift (model trained on 32×32 CIFAR-10 style images).

---

## ⚠️ Important Technical Limitations & Disclaimers

> [!WARNING]
> **Dataset Domain Limitation**: The model was trained strictly on the **CIFAKE dataset** (32×32 native resolution images — CIFAR-10 real photos vs Stable Diffusion v1.4 AI images).
> Modern AI generators (Gemini, Midjourney v6, DALL-E 3, Flux.1, etc.) were **not represented in training**. The advanced Patch, MultiScale, Hybrid, and TTA inference strategies improve real-world robustness without retraining, but this system **does not guarantee generalization to all unseen or future AI generators**.

> [!WARNING]
> **Camera EXIF does not prove image authenticity.** Camera hardware metadata (Make/Model tags) is informational only and does not bypass the PyTorch model verdict. AI edits performed on-device (Pixel Magic Eraser, Samsung AI, iPhone Clean Up) may retain camera EXIF while producing AI-modified content.

---

## 📁 Project Structure

```text
ALGONAUTS_INTERNAL_LJ_HACK-main/
│
├── README.md                               # This file (Entry point)
├── AUDIT_REPORT.md                         # Forensic technical audit report
├── requirements.txt                        # Pinned dependency versions
├── rules.md                                # Team collaboration guidelines
│
├── model/
│   ├── best_resnet50_cifake_retrained.pth  # Trained PyTorch checkpoint (~90MB)
│   └── generator_attribution.py           # Bonus B — Gemini generator family classifier
│
├── app/
│   ├── app.py                              # Streamlit web application (UI)
│   ├── predictor.py                        # Unified facade API
│   ├── config.py                           # Global constants & strategy thresholds
│   ├── model_loader.py                     # ResNet-50 loader + Streamlit cache
│   ├── api/
│   │   ├── gemini_gateway.py               # Gemini gateway with failover, LRU cache, rate limiting
│   │   └── prompt_registry.py              # Versioned prompt contracts (v1)
│   ├── diagnostics/
│   │   ├── entropy.py                      # Shannon entropy + confidence labels
│   │   ├── disagreement.py                 # Patch-level statistical disagreement
│   │   ├── fft_spectral.py                 # 2D FFT spectral anomaly scoring
│   │   ├── explainer.py                    # Bonus A — Gemini explanation caller
│   │   ├── metadata_inspector.py           # Bonus D — C2PA/EXIF/AI metadata inspector
│   │   └── multimodal_consistency.py       # Bonus E — image-caption consistency
│   └── strategies/
│       ├── base_strategy.py                # ABC + output schema validator
│       ├── strategy_registry.py            # Singleton strategy registry
│       ├── auto/                           # Resolution-aware dispatcher
│       ├── hybrid/                         # Resize + Patch + FFT consensus
│       ├── multiscale/                     # 3-branch spatial analysis
│       ├── patch/                          # Native 32×32 crop voting
│       ├── resize/                         # Baseline resize strategy
│       └── tta/                            # 8-view TTA
│
├── evaluation/
│   ├── evaluate_strategies.py              # Strategy benchmark suite
│   ├── evaluate_unseen.py                  # Unseen-generator evaluation
│   ├── adversarial_analysis.py             # Bonus G — FGSM attack analysis
│   └── degradation_analysis.py             # Bonus C — compression robustness
│
├── tests/
│   ├── test_predictor.py                   # Core model + preprocessing tests
│   ├── test_patch_inference.py             # Patch extraction + aggregation tests
│   ├── test_multiscale_inference.py        # MultiScale branch tests
│   ├── test_real_world_stability.py        # Stability with synthetic images
│   └── test_strategy_registry.py          # Registry lookup tests
│
├── docs/                                   # Technical documentation
└── notebook/                               # Google Colab training notebook
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup

```powershell
# Activate virtual environment
.\.venv\Scripts\activate

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Setup Gemini API Key

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
# Optional: override default model
# GEMINI_MODEL=gemini-3.1-flash-lite
```

The key is loaded strictly by `app/api/gemini_gateway.py`. All inference runs fully offline without a key — only Modules A, B, E require Gemini.

### Gemini Gateway Architecture

```text
Module A / Module B / Module E → Central Gemini Gateway → Google GenAI API
                                          ├── Primary: gemini-3.1-flash-lite
                                          ├── Backup 1: gemini-3.5-flash
                                          └── Backup 2: gemini-3.5-flash-lite
```

**Gateway Features:**
1. **Multi-Model Automatic Failover**: Cascades to next model on 429/404 without exposing errors.
2. **LRU Response Cache (500 entries max)**: Thread-safe in-memory caching with bounded eviction.
3. **Zero-Thinking Latency (`thinking_budget=0`)**: Slashes latency from >45s to ~2.5–5s.
4. **Rate Limiting**: 12 req/min rolling window, 1s minimum spacing, bounded exponential backoff.
5. **Request Coalescing**: In-flight deduplication via `threading.Event` prevents redundant API calls.
6. **`st.session_state` Smart Caching**: Tab switching uses 0 additional API calls.

### 3. Run the Application

```powershell
streamlit run app/app.py
```

Opens at `http://localhost:8501`.

### 4. Run Tests

```powershell
.\.venv\Scripts\pytest tests/ -v
```

### 5. Run Bonus Evaluations

```powershell
python evaluation/evaluate_unseen.py
python evaluation/degradation_analysis.py
python evaluation/adversarial_analysis.py
```

---

## 🎥 Demo Video

[Insert Link to 3-5 minute unlisted YouTube demo video here]

---

## 🛡️ User & Safety Disclaimer

*SignalScope is designed as an AI screening and assistance tool. Prediction scores reflect model probabilities under selected inference strategies and should not be used as sole legal or definitive proof of image authenticity. Explanations are provided as likelihood assessments and are not accusations against individuals.*
