# 01. Project Overview — SignalScope

## 1. Problem Statement
The proliferation of generative artificial intelligence models (such as Stable Diffusion, Midjourney, Flux.1, and DALL-E 3) allows synthetic imagery to be created easily and at scale. Distinguishing authentic camera photography from AI-generated synthetic images is critical for media verification, forensic screening, and content moderation.

## 2. Motivation & Objectives
SignalScope was created to provide a lightweight, local, privacy-focused image screening application.
Key objectives:
1. **Local Inference**: Execute AI image classification locally without transmitting images to cloud APIs.
2. **Transparent Computation**: Provide live confidence percentages, class probabilities, raw model output logits, and strategy-level diagnostics.
3. **Reproducible Evaluation**: Establish a verified benchmark pipeline evaluated against standard dataset benchmarks.

---

## 3. Technology Stack
- **Language**: Python 3.11+
- **Deep Learning**: PyTorch 2.14.0 & Torchvision 0.29.0
- **Web UI**: Streamlit 1.63.0
- **Data & Image Processing**: NumPy 2.4.6, Pandas 3.0.5, Pillow 12.3.0
- **Testing & Metrics**: Scikit-Learn 1.9.1, Pytest 9.1.1
- **Gemini Integration**: google-genai 2.23.0
- **Provenance**: c2pa-python 0.37.10 (C2PA Content Credentials)
- **Visualization**: OpenCV 5.0.0 (headless), Matplotlib 3.11.1, Seaborn 0.13.2
- **Environment**: python-dotenv 1.2.3

---

## 4. High-Level Workflow, Inputs & Outputs

```
  ┌───────────────────┐        ┌───────────────────┐        ┌──────────────────────┐
  │  Input Image File │ ─────> │ SignalScope App   │ ─────> │  Prediction Output   │
  │  (JPG, JPEG, PNG) │        │ (ResNet-50 Model) │        │  FAKE / REAL /       │
  └───────────────────┘        └───────────────────┘        │  UNCERTAIN (some     │
                                                             │  strategies)         │
                                                             └──────────────────────┘
```

- **Input**: Single or multiple image files (JPG, JPEG, PNG format).
- **Output**: Binary classification (`FAKE` vs `REAL`), confidence score (%), class probabilities, raw PyTorch logits, strategy diagnostics, and latency timing.

---

## 5. Core Features & Hackathon Modules
- **Core Multi-Strategy Inference**: Six robust strategies including Baseline Resize, Variance-Guided Native Patch Voting (with config-unified Top-K ratio), 3-Branch MultiScale Analysis, 8-View Test-Time Augmentation (TTA), Hybrid Consensus (calibrated decision tree), and an intelligent Resolution-Aware Auto-Dispatcher.
- **Bonus A (Faithful Explanation)**: Grad-CAM heat-maps layered with Gemini-generated natural language cues. Passes FFT, entropy, patch agreement, branch disagreement and strategy mode as structured diagnostic context.
- **Bonus B (Generator Attribution)**: Offline attribution of specific AI generators via `c2pa-python` byte-decoding. Expanded generator list covers Flux.1, Adobe Firefly, Imagen 3, Sora, Kling, Runway, Pika, Ideogram, Canva AI and more. Falls back to Gemini multimodal analysis with neutral framing.
- **Bonus C (Robustness)**: High resilience to degradation (compression/resize) via `PATCH_N=32` stable sampling.
- **Bonus D (Provenance)**: EXIF and C2PA binary signature extraction (44 known AI signatures). CAMERA_REAL is informational only — does not bypass the PyTorch model verdict.
- **Bonus E (Multimodal)**: Image-caption consistency validation JSON scoring.
- **Bonus F (Deployable UI)**: Real-time interactive Streamlit web dashboard with live diagnostics, bounding box annotation, and Responsible AI framing.
- **Bonus G (Active Defence)**: FGSM adversarial attack robustness analysis documented separately.
- **Pretrained Checkpoint**: Loads pre-trained weights (`model/best_resnet50_cifake_retrained.pth`) without retraining during application execution.

---

## 6. Scope & High-Level Limitations

> [!IMPORTANT]
> **Benchmark Classification vs. Universal Detection**: SignalScope evaluates images based on artifact patterns learned from the **CIFAKE dataset** (32×32 CIFAR-10 images vs Stable Diffusion v1.4). It is designed as an assisted screening tool and **does not guarantee universal detection** across unrepresented generative architectures (e.g. Midjourney v6, Flux.1, DALL-E 3, Sora) or modern high-resolution webcam photographs.

> [!NOTE]
> **Camera EXIF does not prove authenticity.** Camera hardware metadata is informational only and does not override the neural network verdict. AI edits performed on-device (Pixel Magic Eraser, Samsung AI, iPhone Clean Up) may retain camera EXIF while producing AI-modified content.
