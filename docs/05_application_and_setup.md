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
- **Model & Session Caching**: Uses `@st.cache_resource` in `app/model_loader.py` to cache the 94.3 MB ResNet-50 checkpoint, and `st.session_state` in `app/app.py` to cache inference and Gemini multimodal responses per image. Tab switching, slider changes, and diagnostic inspection consume **0 additional API calls** and render with 0ms latency.
- **On-Demand Retry Controls**: Dedicated "🔄 Retry" buttons in the UI for Generator Attribution and Faithful Explanation allow immediate recovery from transient rate limits without re-uploading the image.

---

## 2. Step-by-Step Inference Pipeline

Every input image undergoes the following inference sequence:

1. **File Upload**: Image stream received via Streamlit file uploader.
2. **File Validation**: `uploaded_file.seek(0)` resets byte stream pointer.
3. **PIL Decoding**: `Image.open(uploaded_file)` decodes image file.
4. **EXIF Correction**: `ImageOps.exif_transpose(image)` auto-rotates camera metadata tags.
5. **RGB Standardization**: `image.convert("RGB")` converts Grayscale or RGBA images to 3-channel RGB.
6. **Metadata Pre-Screening**: `metadata_inspector.inspect_image_metadata()` scans EXIF, PNG info, and raw bytes for C2PA manifests and AI signatures (44 known signatures). Result is informational — does not bypass PyTorch model except for confirmed `AI_GENERATED` C2PA verdicts.
7. **Strategy Dispatch** (`auto` mode routing by minimum image dimension):
   - `< 64px` → Resize (resize to 32×32 → forward pass)
   - `64–255px` → Patch (extract native 32×32 crops → batch forward pass → vote)
   - `256–511px` → Hybrid (Resize + Patch + FFT spectral decision tree)
   - `≥ 512px` → MultiScale (Global + Context + Native Texture 3-branch fusion)
8. **Tensor Conversion**: `transforms.ToTensor()` scales pixel values $[0, 255] \rightarrow [0.0, 1.0]$.
9. **Normalization**: `transforms.Normalize()` applies ImageNet Z-scores.
10. **Inference**: `torch.inference_mode()` forward pass. Model outputs raw logits $z_0$ (FAKE) and $z_1$ (REAL).
11. **Softmax Activation**: `F.softmax(logits, dim=1)` calculates class probabilities.
12. **Strategy Fusion** (Hybrid / MultiScale): Decision tree or weighted branch fusion produces final probability.
13. **Diagnostics**: Entropy, disagreement, FFT spectral score computed.
14. **Prediction Format**: Returns structured dictionary with label, confidence, probabilities, diagnostics, and inference metadata.

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
- `MODEL_PATH`: `Path` to `model/best_resnet50_cifake_retrained.pth`.
- `CLASS_MAPPING`: `{0: "FAKE", 1: "REAL"}`.
- `IMAGE_SIZE`: `(32, 32)`.
- `IMAGENET_MEAN` / `IMAGENET_STD`: Standard normalization vectors.
- `BENCHMARK_METRICS`: Benchmark performance constants table.

### `app/model_loader.py`
- `get_device() -> torch.device`: Returns `cuda` if available, else `cpu`.
- `load_model(model_path=None) -> tuple[nn.Module, torch.device]`: Instantiates ResNet-50, auto-detects checkpoint stem variant (standard `7×7` vs CIFAR-adapted `3×3`), restores weights, sets `eval()` mode, and caches model via `@st.cache_resource` (Streamlit) or module-level dict (tests/scripts).

### `app/predictor.py`
- `preprocess_image(image: Image.Image) -> torch.Tensor`: Preprocesses image into tensor shape `(1, 3, 32, 32)`.
- `predict_image(...) -> dict`: Facade method that delegates to the appropriate strategy.
- `predict_batch(...) -> pd.DataFrame`: Runs inference on multiple images safely handling exceptions.

### `app/strategies/`
- **`base_strategy.py`**: Abstract base class + `validate_strategy_output()` schema validator.
- **`strategy_registry.py`**: Singleton registry — lists and looks up strategies by name.
- **`auto_strategy.py`**: 4-tier dispatcher routing by minimum image dimension (`<64px` → resize, `64-255px` → patch, `256-511px` → hybrid, `≥512px` → multiscale). FFT is only computed when hybrid is selected.
- **`hybrid_strategy.py`**: Decision tree combining Resize, Patch, and FFT diagnostics. Calibrated Branch 1b uses threshold-based label (not hardcoded REAL).
- **`multiscale_strategy.py`**: 3-branch spatial analysis (Global resize + 16 Context patches + N Native Texture crops) with adaptive weight fusion. Supports tri-state output: REAL / FAKE / UNCERTAIN.
- **`patch_strategy.py`**: Variance-guided native 32×32 crop extraction. Top-K ratio (20%) consistent with `MULTISCALE_TOP_K_RATIO` config constant. Multiple aggregation modes: `mean`, `median`, `majority`, `logit_mean`, `max`, `top_k`.
- **`tta_strategy.py`**: 8-view augmentation (Original, H-Flip, Center Crop, Bright±15%, Contrast+20%, Rotate 90°/45°) with inverse-entropy weighted aggregation.

### `app/diagnostics/` (Bonus Modules)
- **`entropy.py`**: Shannon entropy ($H/\ln2$) + confidence label bands.
- **`disagreement.py`**: Patch-level statistical disagreement metrics (mean, std, range, agreement %).
- **`fft_spectral.py`**: 2D FFT radial power spectrum scoring. Used exclusively by hybrid strategy.
- **`grad_cam.py`**: Gradient-weighted Class Activation Mapping heatmap overlay.
- **`bounding_box.py`**: Top-N suspect region RGBA overlay renderer.
- **`explainer.py`**: Gemini faithful explanation caller with structured diagnostic context (Module A).
- **`metadata_inspector.py`**: C2PA/EXIF/PNG AI provenance inspector with 44 known AI signatures. CAMERA_REAL is informational only — does not bypass model (Module D).
- **`multimodal_consistency.py`**: Image-caption consistency JSON scorer (Module E).

### `model/generator_attribution.py` (Bonus B Gemini fallback)
Gemini multimodal classifier for generator family attribution. Uses `app.api` imports. Located in `model/` for historical reasons but architecturally belongs with the diagnostics package.

### `app/api/` (Centralized Gemini Integration)
- **`gemini_gateway.py`**: Central gateway implementing primary `gemini-3.1-flash-lite`, multi-model failover, zero-thinking latency (`thinking_budget=0`), thread-safe LRU cache (500 entries max), local rate limiting (12 RPM), request coalescing, bounded 429 retries, and regex markdown fence stripping.
- **`prompt_registry.py`**: Versioned prompt contracts. Generator attribution uses neutral framing with 30+ generator families. Faithful explanation uses 300-token budget. Multimodal consistency scoped to visible evidence only.

### `app/app.py`
Streamlit application entry point implementing header, tabs, single-image preview, prediction visual boxes, live diagnostic logits expander, bounding box visualization, batch upload, and CSV downloads. Integrated with `st.session_state` smart caching and retry controls.
