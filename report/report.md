# SignalScope - One Page Model Report

**Team:** ALGONAUTS (C-433)
**Hackathon:** SIH - 2026

### 1. Task
- **Core:** Binary real-vs-AI-generated image classification.
- **Bonus Modules Attempted:** 
  - A (Faithful Explanation): Human-readable textual explanations of visual cues via Gemini 2.5 API.
  - B (Generator Attribution): Multi-class attribution (Diffusion vs GAN) via Gemini.
  - C (Robustness to Degradation): Empirical analysis against JPEG compression and resizing.
  - D (Provenance & Metadata): C2PA / Content Credentials and EXIF pre-screening.
  - E (Multimodal): Image-text consistency scoring via Gemini 2.5 API.
  - F (Real-Time / Deployable): Streamlit Web UI with drag-and-drop & batch scanning.
  - G (Active Defence Analysis): FGSM adversarial attack failure analysis.

### 2. Data & split
- **Core Training Data:** CIFAKE Dataset (MIT Licensed, ~100k+ labelled $32 \times 32$ images, balanced real/fake).
- **Split:** 80% Train, 10% Validation, 10% Test (Original CIFAKE standard split). No additional public data was mixed during training.
- **Evaluation:** Evaluated on both the standard CIFAKE test set and a synthetic mock dataset for "unseen generator" (Midjourney-style patterns).

### 3. Model / approach
- **Backbone:** ResNet-50 deep learning model, fine-tuned with a customized $32 \times 32$ input stem.
- **Key Hyperparameters:** Evaluated over 8, 16, 32, and 64 patches per image. Patch Aggregation using Mean / Median Probability.
- **Augmentation & Robustness:** Implemented Test-Time Augmentation (TTA) with 8 geometrical/photometric views, and Native-Resolution Patch Voting to avoid downscaling destruction of AI artifacts.
- **Calibration:** Uses Normalized Shannon Entropy to communicate prediction uncertainty.

### 4. Metric & result
*Note: Evaluated on the held-out test sets.*
- **Overall AUC (CIFAKE):** 0.9987
- **Unseen-Split AUC (Mock Dataset):** ~0.95+ (See /evaluation/unseen_generator_metrics.csv)
- **Macro-F1 (CIFAKE):** 0.9832
- **Accuracy (CIFAKE):** 98.33%
- **False-Positive Rate:** 1.69% at 0.5 threshold.
- **Confusion Matrix:** True Positive: 9834, True Negative: 9831, False Positive: 169, False Negative: 166.

### 5. Baseline
The provided baseline model is expected to suffer heavily on the unseen-split. By using the Native Patch Voting + Hybrid Consensus (ResNet-50), our system filters out localized artifacts, preventing catastrophic failure on unseen generators. The Hybrid architecture specifically targets high-frequency synthetic generator patterns (verified by the experimental 2D FFT spectral diagnostics) giving a significant edge over standard global-resize baselines.

### 6. Limitations
- **Adversarial Vulnerability:** The model is highly susceptible to FGSM attacks (demonstrated in Active Defence Analysis); even a small $\epsilon=0.05$ causes a steep accuracy drop.
- **Unseen Generator Caveat:** Though robust against minor perturbations, the ResNet-50 was solely trained on CIFAKE (Stable Diffusion 1.4 artifacts). Extremely modern generators (like FLUX or Midjourney v6) might bypass detection if they lack the specific localized frequency artifacts our patch-vote system looks for.
