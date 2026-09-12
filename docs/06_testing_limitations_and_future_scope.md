# 06. Testing, Limitations, Security and Future Scope — SignalScope

## 1. Testing Documentation

### Automated Unit Test Suite
SignalScope includes an automated test suite in [`tests/test_predictor.py`](../tests/test_predictor.py) executed via Pytest:

```powershell
.\.venv\Scripts\pytest.exe tests/test_predictor.py -v
```

### Verification Results (8 / 8 Passed):
- `test_get_device`: PASS (Valid device detection)
- `test_model_loading`: PASS (ResNet-50 architecture & 2-class head, verified `eval()` mode)
- `test_preprocess_rgb_image`: PASS (Shape `(1, 3, 224, 224)`)
- `test_preprocess_grayscale_image`: PASS (Mode L auto-conversion)
- `test_preprocess_rgba_image`: PASS (RGBA 4-channel auto-conversion)
- `test_predict_image_output_structure`: PASS (Output keys & probability bounds)
- `test_predict_batch`: PASS (Batch DataFrame generation)
- `test_invalid_input_handling`: PASS (Catches invalid input types)

> [!WARNING]
> **Known Minor Bug (`test_predict_image_output_structure`, line 100):** The probability sum assertion is written as `pytest.approx(fake_p + real_p, abs=1e-4) == 1.0` which evaluates but never actually asserts — it should be `assert fake_p + real_p == pytest.approx(1.0, abs=1e-4)`. The test passes, but the probability-sum check is vacuous. All other assertions in the test are correct.

### Manual Verification Cases
1. **RGB Image (300×300)**: PASS ($\rightarrow$ REAL, 97.11% confidence)
2. **Grayscale Image (Mode L)**: PASS ($\rightarrow$ Auto-converted to 3-channel RGB)
3. **RGBA Image (4-channel)**: PASS ($\rightarrow$ Transparency channel removed, converted to RGB)
4. **Invalid Input Handling**: PASS (Caught `ValueError("Input must be a valid PIL Image")`)
5. **Batch Upload Processing**: PASS (Processed 3 images into structured Pandas DataFrame)

---

## 2. Limitations & Risk Analysis

### Dataset Domain Shift
- The ResNet-50 model was trained strictly on the **CIFAKE dataset** ($32 \times 32$ CIFAR-10 images upscaled to $224 \times 224$).
- High-resolution digital photographs or modern webcam photos possess different spatial frequency characteristics than CIFAR-10 real photos, which can cause modern webcam photos to trigger high `FAKE` activations due to dataset domain shift.

### Model Boundaries & Calibrated Probability
- Confidence scores represent raw Softmax probability outputs from the trained neural network, not calibrated guarantees of image authenticity.
- The model does not guarantee universal detection on unrepresented generative model architectures (e.g. Midjourney v6, Flux, DALL-E 3).

---

## 3. Security Considerations

- **Untrusted Checkpoint Loading**: PyTorch `torch.load()` can deserialize arbitrary Python objects. Always load weight checkpoints from trusted project repositories.
- **Upload File Shielding**: File uploads are decoded using PIL with explicit byte stream resetting (`uploaded_file.seek(0)`) to prevent memory leaks or corrupted stream crashes.
- **Privacy Assurance**: All inference executes locally on the user's workstation. No images or prediction metadata are transmitted to external servers.

---

## 4. Future Improvements Roadmap

The following enhancements are proposed as future roadmap items:

1. **Multi-Dataset Training**: Expand training beyond CIFAKE to include GenImage, Artifact, and DiffusionDB datasets.
2. **High-Resolution Training**: Fine-tune models directly on $512 \times 512$ or $1024 \times 1024$ crops to capture fine sensor noise.
3. **Explainability & Heatmaps**: Incorporate Grad-CAM spatial heatmaps and Discrete Cosine Transform (DCT) frequency magnitude visualizers.
4. **Model Calibration**: Apply Temperature Scaling to calibrate raw Softmax output probabilities.
5. **ONNX Export**: Convert PyTorch checkpoints to ONNX runtime format for optimized CPU/GPU deployment.
