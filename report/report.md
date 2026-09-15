# SignalScope - One Page Model Report

**Team:** ALGONAUTS (C-433)
**Hackathon:** SIH - 2026

### 1. Task
- **Core:** Binary real-vs-AI-generated image classification.
- **Bonus Modules Attempted:** 
  - A (Faithful Explanation): Textual explanations of visual cues via Gemini API (`gemini-3.1-flash-lite` with multi-model failover).
  - B (Generator Attribution): Precise model extraction (e.g., Adobe Firefly) via `c2pa-python` binary manifest decoding, plus multi-class attribution via Gemini API.
  - C (Robustness to Degradation): High resilience to resizing and compression through `PATCH_N=32` stable patch consensus rather than global downscaling.
  - D (Provenance & Metadata): Native byte scanning for C2PA `jumbc2pa` manifests, EXIF hardware tags, and Messenger artifacts.
  - E (Multimodal): Image-caption consistency validation scoring via Gemini API.
  - F (Real-Time / Deployable): Real-time Streamlit Web UI with responsible "Provenance Verdict" framing, drag-and-drop, and batch scanning.
  - G (Active Defence Analysis): Thorough FGSM adversarial attack failure analysis and degradation testing.

### 2. Data & split
- **Core Training Data:** CIFAKE Dataset (MIT Licensed, ~100k+ labelled $32 \times 32$ images, balanced real/fake).
- **Split:** 80k Train, 20k Validation, 20k Test (Original CIFAKE standard split). No additional public data was mixed during training.
- **Evaluation:** Evaluated on both the standard CIFAKE test set and a synthetic mock dataset for "unseen generator" (Midjourney-style patterns).

### 3. Model / approach
- **Backbone:** ResNet-50 deep learning model, fine-tuned with a customized $32 \times 32$ input stem.
- **Key Hyperparameters:** Fixed stability parameter `PATCH_N = 32` patches per image to prevent dynamic scaling noise. Patch Aggregation relies on a capped 70% confidence majority threshold.
- **Augmentation & Robustness:** Implemented Test-Time Augmentation (TTA) with 8 geometrical/photometric views, and Native-Resolution Patch Voting to avoid downscaling destruction of AI artifacts.
- **Calibration:** Uses Normalized Shannon Entropy to communicate prediction uncertainty.

### 4. Metric & result
*Note: Evaluated on the held-out test sets.*
- **Overall AUC (CIFAKE):** 0.9951
- **Unseen-Split AUC (Mock Dataset):** ~0.95+ (See /evaluation/unseen_generator_metrics.csv)
- **Macro-F1 (CIFAKE):** 0.9677
- **Accuracy (CIFAKE):** 96.77%
- **False-Positive Rate:** 2.95% at 0.608 threshold.
- **Confusion Matrix:** True Positive: 9649, True Negative: 9705, False Positive: 295, False Negative: 351.

### 5. Baseline
The provided baseline model is expected to suffer heavily on the unseen-split. By using the Native Patch Voting + Hybrid Consensus (ResNet-50), our system filters out localized artifacts, preventing catastrophic failure on unseen generators. The Hybrid architecture specifically targets high-frequency synthetic generator patterns (verified by the experimental 2D FFT spectral diagnostics) giving a significant edge over standard global-resize baselines.

### 6. Limitations
- **Adversarial Vulnerability:** The model is highly susceptible to FGSM attacks (demonstrated in Active Defence Analysis); even a small $\epsilon=0.05$ causes a steep accuracy drop.
- **Unseen Generator Caveat:** Though robust against minor perturbations, the ResNet-50 was solely trained on CIFAKE (Stable Diffusion 1.4 artifacts). Extremely modern generators (like FLUX or Midjourney v6) might bypass detection if they lack the specific localized frequency artifacts our patch-vote system looks for.
