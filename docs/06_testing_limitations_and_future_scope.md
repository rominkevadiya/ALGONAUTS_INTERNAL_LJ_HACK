# 06. Testing, Limitations, Security and Future Scope — SignalScope

## 1. Testing Documentation

### Automated Unit Test Suite

SignalScope includes an automated test suite executed via Pytest:

```powershell
.\.venv\Scripts\pytest tests/ -v
```

### Test Files & Coverage (39 / 39 Passed)

| Test File | What It Covers | Tests |
| :--- | :--- | :---: |
| `test_predictor.py` | Device detection, model loading, preprocessing (RGB/Grayscale/RGBA), single-image inference output schema, batch prediction DataFrame, invalid input handling | 8 |
| `test_patch_inference.py` | Image mode handling, large image patch inference, small image fallback, patch count correctness, deterministic coordinates, aggregation methods, entropy, FFT diagnostic, hybrid/TTA/auto output | 13 |
| `test_multiscale_inference.py` | Frozen model integrity, object-context 16-patch extraction, native texture extraction, tiny image fallback, branch statistics aggregation, full multiscale pipeline | 6 |
| `test_real_world_stability.py` | Stability across patch counts, stability across seeds, real photos not falsely flagged | 3 |
| `test_strategy_registry.py` | Registry listing, output schema validation, multiscale/auto/patch/resize run via registry | 5 |
| **Total** | | **39** |

### Manual Verification Cases
1. **RGB Image (300×300)**: PASS (→ REAL, high confidence)
2. **Grayscale Image (Mode L)**: PASS (→ Auto-converted to 3-channel RGB)
3. **RGBA Image (4-channel)**: PASS (→ Transparency channel removed, converted to RGB)
4. **Invalid Input Handling**: PASS (Caught `ValueError`)
5. **Batch Upload Processing**: PASS (Processed multiple images into structured Pandas DataFrame)
6. **Large image (≥512px) auto route**: Routes to `multiscale` (verified via terminal log)
7. **Medium image (256–511px) auto route**: Routes to `hybrid`

---

## 2. Limitations & Risk Analysis

### Dataset Domain Shift (Primary Limitation)
- The ResNet-50 model was trained strictly on the **CIFAKE dataset** (32×32 CIFAR-10 images upscaled to 224×224).
- High-resolution digital photographs or modern AI outputs (Midjourney v6, Flux.1, DALL-E 3, Sora) possess different spatial frequency characteristics than CIFAKE, which can degrade classification accuracy.
- The MultiScale, Hybrid, and Patch strategies mitigate this through native-resolution crop analysis, but cannot fully compensate for training distribution mismatch.

### Hybrid Strategy Decision Tree
- The hybrid strategy uses 11 manually tuned threshold values (e.g., 0.92, 0.65, 0.45) determined through empirical observation on a small image set. These may not generalize to all image types. UNCERTAIN outputs are not supported in hybrid mode.

### Model Boundaries & Calibrated Probability
- Confidence scores represent raw Softmax probability outputs from the trained neural network, not calibrated guarantees of image authenticity.
- The model does not guarantee universal detection on unrepresented generative model architectures.

### Metadata Inspector Limitations
- EXIF metadata can be forged. A real photo with a fabricated "midjourney" tag will trigger `AI_GENERATED`. Camera EXIF does not prove authenticity — it is treated as informational only and does not bypass the neural network verdict.
- C2PA manifests can be stripped from JPEG files without altering visible content.

---

## 3. Security Considerations

- **Checkpoint Loading Security**: PyTorch checkpoint is loaded with `weights_only=True` and a comprehensive numpy dtype allowlist (`numpy.dtype`, `numpy._core.multiarray.scalar`, and all `numpy.dtypes.*` concrete types). This prevents arbitrary Python code execution from a malicious `.pth` file while maintaining compatibility with the checkpoint's numpy metadata.
- **Upload File Shielding**: File uploads are decoded using PIL with explicit byte stream resetting (`uploaded_file.seek(0)`) to prevent memory leaks or corrupted stream crashes.
- **Privacy Assurance**: All inference executes locally on the user's workstation. No images or prediction metadata are transmitted to external servers. Gemini API calls transmit only the image + structured diagnostic context (no PII).
- **API Key Security**: Gemini API key is loaded strictly from `.env` via `python-dotenv`. The `.env` file is excluded by `.gitignore`. Never hardcode API keys.

---

## 4. Future Improvements Roadmap

The following enhancements are proposed as future roadmap items (all without retraining):

1. **Real-World Evaluation Dataset**: Collect 100+ labeled images per class from Midjourney v6, Flux.1, DALL-E 3, and authentic camera photos. Run `evaluate_strategies.py --dataset <path>` to generate real-world accuracy benchmarks per strategy.
2. **Hybrid Threshold Calibration**: Replace the 11 manually-tuned thresholds in `hybrid_strategy.py` with empirically derived constants from a labeled real-world evaluation set.
3. **UNCERTAIN Support in Hybrid**: Add tri-state output to hybrid strategy for ambiguous cases (e.g., 51% fake vs 49% real disagreement between resize and patch).
4. **Model Calibration**: Apply Temperature Scaling to calibrate raw Softmax output probabilities into reliable confidence estimates.
5. **ONNX Export**: Convert PyTorch checkpoint to ONNX runtime format for optimized CPU/GPU deployment and reduced startup time.
6. **Inference Timeout**: Add per-strategy execution timeout (e.g., 30s) to prevent indefinite UI blocking on large images with `concurrent.futures`.
7. **Image Size Guard**: Add upload file size validation (e.g., 20MB max) to prevent memory exhaustion on extremely large images.
8. **Multi-Dataset Training** *(requires retraining)*: Expand training beyond CIFAKE to include GenImage, Artifact, and DiffusionDB datasets for broader generator coverage.
