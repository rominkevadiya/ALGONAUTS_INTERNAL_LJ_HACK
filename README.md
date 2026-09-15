# SignalScope 🔍

SignalScope is an AI-based screening tool for distinguishing authentic camera photography from AI-generated synthetic images locally and transparently.

## 1. Implemented Modules

| Module | Status | Implementation Details |
| :--- | :---: | :--- |
| **Core: Multi-Strategy Inference** | ✅ | Implemented 6 strategies (`resize`, `patch`, `multiscale`, `tta`, `hybrid`, `auto`). |
| **Bonus A: Faithful Explanation** | ✅ | Implemented via Gemini multimodal analysis with diagnostic context (FFT, entropy). |
| **Bonus B: Generator Attribution** | ✅ | Implemented via `c2pa-python` byte-decoding (30+ known AI signatures). |
| **Bonus C: Robustness** | ✅ | Implemented via `PATCH_N=32` stable sampling against degradation. |
| **Bonus D: Provenance** | ✅ | Implemented via EXIF and C2PA binary signature extraction. |
| **Bonus E: Multimodal** | ✅ | Implemented via Image-Caption consistency JSON scoring. |
| **Bonus F: Deployable UI** | ✅ | Implemented via real-time Streamlit dashboard. |
| **Bonus G: Active Defence** | ✅ | Documented FGSM adversarial attack robustness analysis. |

## 2. Setup and Run Instructions (Reproducibility)

Follow these steps to reproduce a prediction in under 10 minutes:

```bash
# 1. Clone the repository
git clone https://github.com/rominkevadiya/ALGONAUTS_INTERNAL_LJ_HACK.git
cd ALGONAUTS_INTERNAL_LJ_HACK

# 2. Create and activate a virtual environment
python3 -m venv .venv
# On Windows: .\.venv\Scripts\activate
# On Mac/Linux: source .venv/bin/activate

# 3. Install exactly pinned requirements
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Optional) Set up Gemini API key in a .env file for Bonus A & E
echo "GEMINI_API_KEY=your_key_here" > .env

# 5. Run the web application
streamlit run app/app.py
```

## 3. Datasets Used

- **Dataset Name**: CIFAKE: Real and AI-Generated Synthetic Images
- **Source**: Kaggle (`birdy666/cifake-real-and-ai-generated-synthetic-images`)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0) (Assumed standard for this public Kaggle dataset)
- **Citation**: Bird, J. J., & Lotfi, A. (2023). *CIFAKE: Real and AI-Generated Synthetic Images*. Kaggle.
- *Note: No additional external datasets were merged into the core training pipeline.*

## 4. Evaluation Metrics

Metrics were pulled directly from empirical testing (`evaluation/final_evaluation_metrics.csv` and `evaluation/unseen_generator_metrics.csv`). 

| Metric | Score |
| :--- | :--- |
| **Overall Accuracy** | 98.33% (`0.98325`) |
| **Macro-F1 Score** | `0.9832` |
| **ROC-AUC** | `0.9987` |
| **PR-AUC** | `0.9988` |
| **Confusion Matrix** | See `outputs/confusion_matrix.png` |
| **Unseen-Generator Accuracy** | 100.00% |
| **Unseen-Generator ROC-AUC** | 1.0000 |

## 5. Architecture Overview

- **Backbone**: PyTorch ResNet-50 (`23,565,303` parameters).
- **Input Pipeline / Stem**: The standard 7×7 ImageNet stem is replaced by a CIFAR-adapted 32×32 stem (`Conv2d(3, 64, kernel_size=3, stride=1, padding=1)` and `nn.Identity()`).
- **Checkpoint Location**: `model/best_resnet50_cifake_retrained.pth` (~90 MB).
- **Predict Interface**: External graders can evaluate the model programmatically via `app.predictor`:
  ```python
  from app.predictor import predict_image_auto

  result = predict_image_auto(
      image=pil_image,
      model=loaded_model,
      device="cpu",
      mode="auto",           # 'auto', 'multiscale', 'hybrid', 'resize', 'patch', 'tta'
      n_patches=32,
      seed=42,
      aggregation="mean",
      precomputed_metadata=None,
      raw_bytes=image_bytes
  )
  ```
  **Output Format**: Returns a dictionary containing `label` (String: FAKE/REAL), `confidence` (Float), `fake_prob` (Float), `real_prob` (Float), and internal strategy diagnostic data (entropy, FFT spectral, etc.).

## 6. Honest Limitations

- **Dataset Domain Shift**: The model is trained strictly on the CIFAKE dataset (32×32 images upscaled). High-resolution digital photographs or webcam photos of human faces exhibit a severe domain shift, frequently triggering high activation on Class 0 (`FAKE`).
- **Spoofable EXIF Data**: Camera EXIF metadata does not prove authenticity. AI edits performed on-device (e.g., Pixel Magic Eraser) retain camera EXIF. The model relies on the neural network verdict and treats `CAMERA_REAL` as informational only.
- **Evadable C2PA Signatures**: C2PA signatures and metadata overrides (such as "midjourney") can be easily stripped by users before uploading, requiring fallback to the neural network or Gemini multimodal analysis.

## 7. Links

- **Live Deployed App URL**: [http://52.66.64.204:8501](http://52.66.64.204:8501)
- **Demo Video**: https://drive.google.com/drive/folders/1B1tI9txvvubaSZSI0Awd5PhogTiATkae
- - **Project Report:** [View Detailed Report](report/report.md)
- **Deployment Notes**: Currently hosted on an AWS `t3.micro` EC2 instance with a 4GB Swap file to support PyTorch memory requirements.

## 8. Originality Declaration

This project utilizes the following third-party libraries and code:
- **PyTorch/Torchvision**: Used for the ResNet-50 backbone and pre-trained ImageNet weights (`ResNet50_Weights.DEFAULT`).
- **Streamlit**: Used for the frontend web application UI.
- **c2pa-python**: Utilized for reading C2PA Content Credentials and binary provenance signatures.
- **google-genai**: Utilized for multimodal explanations via Gemini.
- **OpenCV, Scikit-Learn, Pandas, NumPy**: Standard data processing and evaluation libraries.

---
