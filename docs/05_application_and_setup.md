# 05. Application, Inference Pipeline and Setup Guide — SignalScope

## 1. Streamlit Web Application

The frontend interface (`app/app.py`) is built using **Streamlit 1.63+**.

### Key UI Features & Workflow
- **Image Upload & Validation**: Accepts JPG, JPEG, and PNG files via `st.file_uploader()`. Decodes images using PIL and verifies data bytes before rendering.
- **Prediction Status Badges**:
  - 🔴 **Red Alert Box**: Predicted `FAKE` (Class 0 probability $> 50\%$).
  - 🟢 **Green Success Box**: Predicted `REAL` (Class 1 probability $> 50\%$).
  - 🟠 **Amber Neutral Box**: Low confidence prediction ($< 70\%$).
- **Live PyTorch Diagnostics**: Expandable panel displaying raw unrounded output logits ($z_0, z_1$), Softmax probability values, execution device type (`CPU` / `CUDA`), and forward pass timing in milliseconds.
- **Batch Analysis & CSV Export**: Bulk file uploader allowing multiple image predictions and exporting results as `signalscope_batch_predictions.csv`.
- **Model Caching**: Uses `@st.cache_resource` in `app/model_loader.py` to prevent reloading the 94.3 MB model checkpoint on UI reruns.

---

## 2. Step-by-Step Inference Pipeline

Every input image undergoes the following 14-step inference sequence:

1. **File Upload**: Image stream received via Streamlit file uploader.
2. **File Validation**: `uploaded_file.seek(0)` resets byte stream pointer.
3. **PIL Decoding**: `Image.open(uploaded_file)` decodes image file.
4. **EXIF Correction**: `ImageOps.exif_transpose(image)` auto-rotates camera metadata tags.
5. **RGB Standardization**: `image.convert("RGB")` converts Grayscale or RGBA images to 3-channel RGB.
6. **Spatial Resize**: `transforms.Resize((224, 224))` resizes image to fixed spatial dimensions.
7. **Tensor Conversion**: `transforms.ToTensor()` scales pixel values $[0, 255] \rightarrow [0.0, 1.0]$.
8. **Normalization**: `transforms.Normalize()` applies ImageNet Z-scores ($\text{mean}=[0.485, 0.456, 0.406]$, $\text{std}=[0.229, 0.224, 0.225]$).
9. **Batch Dimension**: `.unsqueeze(0)` shapes tensor to $(1, 3, 224, 224)$.
10. **Device Transfer**: `.to(device)` transfers tensor to CPU or CUDA GPU memory.
11. **Inference Context**: Forward pass executed inside `with torch.no_grad():`.
12. **Forward Pass**: Model outputs raw logits $z_0$ (FAKE) and $z_1$ (REAL).
13. **Softmax Activation**: `F.softmax(logits, dim=1)` calculates class probabilities $P_0$ and $P_1$.
14. **Prediction Format**: Returns structured dictionary containing label, confidence, and class probabilities.

> [!NOTE]
> `model.eval()` is set **once** at checkpoint load time, not per request. Batch inference via `predict_batch()` processes images sequentially (one forward pass per image).

---

## 3. Setup & Installation Guide (Windows PowerShell)

### Inference Setup (Running the App Locally)
*Does NOT require downloading the CIFAKE dataset.*

```powershell
# 1. Navigate to project root
cd d:\ALGONAUTS_INTERNAL_LJ_HACK-main

# 2. Create Python virtual environment
python -m venv .venv

# 3. Activate virtual environment
.\.venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Set up your Gemini API Key for Explainability (Create a .env file in root)
# GEMINI_API_KEY=your_key_here

# 6. Launch Streamlit app
streamlit run app/app.py

# 7. Run unit test suite
pytest tests/test_predictor.py -v
```

### Training Setup (For Model Retraining / Reproduction)
- Requires GPU environment (e.g. Google Colab Tesla T4 GPU).
- Requires downloading the CIFAKE dataset via KaggleHub (`birdy666/cifake-real-and-ai-generated-synthetic-images`).

---

## 4. Module API Reference

### `app/config.py`
- `MODEL_PATH`: `Path` to `model/best_resnet50_cifake_native32_2.pth`.
- `CLASS_MAPPING`: `{0: "FAKE", 1: "REAL"}`.
- `IMAGE_SIZE`: `(224, 224)`.
- `IMAGENET_MEAN` / `IMAGENET_STD`: Standard normalization vectors.
- `BENCHMARK_METRICS`: Benchmark performance constants table.

### `app/model_loader.py`
- `get_device() -> torch.device`: Returns `cuda` if available, else `cpu`.
- `load_model(model_path=None) -> tuple[nn.Module, torch.device]`: Instantiates ResNet-50, auto-detects checkpoint stem variant (standard `7×7` vs CIFAR-adapted `3×3`), restores weights, sets `eval()` mode, and caches model via `@st.cache_resource` (Streamlit) or module-level dict (tests/scripts).

### `app/predictor.py`
- `preprocess_image(image: Image.Image) -> torch.Tensor`: Preprocesses image into tensor shape `(1, 3, 224, 224)`.
- `predict_image(...) -> dict`: Facade method that delegates to the appropriate strategy.
- `predict_batch(...) -> pd.DataFrame`: Runs inference on multiple images safely handling exceptions.

### `app/strategies/`
- **`auto_strategy.py`**: Graduated dispatcher routing based on resolution (`<64px`, `64-256px`, `>256px`).
- **`hybrid_strategy.py`**: Decision tree combining Resize, Patch, and FFT diagnostics.
- **`patch_strategy.py`**: Variance-guided native crop extraction with adaptive luminance thresholds.
- **`tta_strategy.py`**: 8-view geometric/photometric augmentation with inverse-entropy weighting.

### `app/diagnostics/` & `model/` (Bonus Modules)
- **`explainer.py`**: Integrates Gemini API for human-readable faithful explanations and image-text consistency scoring.
- **`metadata_inspector.py`**: Extracts EXIF data and validates C2PA Content Credentials for digital provenance.
- **`grad_cam.py`**: Generates gradient-weighted class activation mapping (Grad-CAM) heatmaps to visualize ResNet focus.
- **`model/generator_attribution.py`**: Uses Gemini API to deduce the exact generator family (e.g. Midjourney vs DALL-E) from visual artifacts.

### `app/app.py`
- Streamlit application entry point implementing header, tabs, single-image preview, prediction visual boxes, live diagnostic logits expander, batch upload, and CSV downloads.
