# 01. Project Overview — SignalScope

## 1. Problem Statement
The proliferation of generative artificial intelligence models (such as Stable Diffusion, Midjourney, and DALL-E) allows synthetic imagery to be created easily. Distinguishing authentic camera photography from AI-generated synthetic images is critical for media verification, forensic screening, and content moderation.

## 2. Motivation & Objectives
SignalScope was created to provide a lightweight, local, privacy-focused image screening application.
Key objectives:
1. **Local Inference**: Execute AI image classification locally without transmitting images to cloud APIs.
2. **Transparent Computation**: Provide live confidence percentages, class probabilities, and raw model output logits.
3. **Reproducible Evaluation**: Establish a verified benchmark pipeline evaluated against standard dataset benchmarks.

---

## 3. Technology Stack
- **Language**: Python 3.11+
- **Deep Learning**: PyTorch & Torchvision
- **Web UI**: Streamlit 1.63+
- **Data & Image Processing**: NumPy, Pandas, Pillow
- **Testing & Metrics**: Scikit-Learn, Pytest

---

## 4. High-Level Workflow, Inputs & Outputs

```
  ┌───────────────────┐        ┌───────────────────┐        ┌───────────────────┐
  │  Input Image File │ ─────> │ SignalScope App   │ ─────> │ Prediction Output │
  │  (JPG, JPEG, PNG) │        │ (ResNet-50 Model) │        │ (FAKE / REAL)     │
  └───────────────────┘        └───────────────────┘        └───────────────────┘
```

- **Input**: Single or multiple image files (JPG, JPEG, PNG format).
- **Output**: Binary classification (`FAKE` vs `REAL`), confidence score (%), class probabilities, raw PyTorch logits, and latency timing.

---

## 5. Core Features & Hackathon Modules
- **Core Multi-Strategy Inference**: Five robust strategies including Baseline Resize, Variance-Guided Native Patch Voting, 8-View Test-Time Augmentation (TTA), Hybrid Consensus, and an intelligent Auto-Dispatcher.
- **Bonus A (Faithful Explanation)**: Grad-CAM heat-maps layered with Gemini-generated natural language cues.
- **Bonus B (Generator Attribution)**: Offline attribution of specific AI generators via `c2pa-python` byte-decoding, with Gemini multi-class fallbacks.
- **Bonus C (Robustness)**: High resilience to degradation (compression/resize) via `PATCH_N=32` stable sampling.
- **Bonus D (Provenance)**: EXIF and C2PA binary signature extraction pre-screening.
- **Bonus E (Multimodal)**: Image-caption consistency validation JSON scoring.
- **Bonus F (Deployable UI)**: Real-time interactive Streamlit web dashboard with live diagnostics and Responsible AI framing.
- **Bonus G (Active Defence)**: FGSM adversarial attack robustness analysis documented separately.
- **Pretrained Checkpoint**: Loads pre-trained weights (`model/best_resnet50_cifake_retrained.pth`) without retraining during application execution.

---

## 6. Scope & High-Level Limitations

> [!IMPORTANT]
> **Benchmark Classification vs. Universal Detection**: SignalScope evaluates images based on artifact patterns learned from the **CIFAKE dataset**. It is designed as an assisted screening tool and **does not guarantee universal detection** across unrepresented generative architectures (e.g. Midjourney v6, Flux, DALL-E 3) or modern high-resolution webcam photographs.
