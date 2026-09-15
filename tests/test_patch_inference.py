"""
Unit Tests for Non-Retraining SignalScope Inference Engine
Tests: Patch Voting, Aggregation Methods, TTA, Hybrid Mode, Entropy, Disagreement, FFT Diagnostic, and Image Normalization.
"""

from pathlib import Path
import pytest
from PIL import Image
import torch
import torch.nn as nn

from app.config import IMAGE_SIZE, PATCH_N, PATCH_AGGREGATION_METHODS
from app.predictor import (
    prepare_image,
    preprocess_image,
    extract_native_patches,
    compute_prediction_entropy,
    compute_prediction_disagreement,
    compute_fft_spectral_diagnostic,
    predict_image,
    predict_image_patch_vote,
    predict_image_tta,
    predict_image_hybrid,
    predict_image_auto,
    predict_batch
)


class DummyModel(nn.Module):
    """
    Lightweight deterministic PyTorch dummy model for fast unit testing.
    Outputs constant logits yielding deterministic probabilities.
    """
    def __init__(self, fake_logit: float = 1.0, real_logit: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(32, 2)
        # Constant bias for deterministic outputs
        with torch.no_grad():
            self.fc.bias[0] = fake_logit
            self.fc.bias[1] = real_logit
            self.fc.weight.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input and run dummy linear pass
        batch_size = x.shape[0]
        out = torch.zeros((batch_size, 2), device=x.device)
        out[:, 0] = self.fc.bias[0]
        out[:, 1] = self.fc.bias[1]
        return out


@pytest.fixture
def dummy_model_and_device():
    model = DummyModel(fake_logit=1.5, real_logit=0.5)
    model.eval()
    device = torch.device("cpu")
    return model, device


def test_prepare_image_modes():
    """1. Tests image preparation helper for RGB, RGBA, LA, P with transparency, and L."""
    # RGB
    img_rgb = Image.new("RGB", (100, 100), color=(100, 150, 200))
    prep_rgb = prepare_image(img_rgb)
    assert prep_rgb.mode == "RGB"
    assert prep_rgb.size == (100, 100)

    # RGBA with transparency
    img_rgba = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    prep_rgba = prepare_image(img_rgba)
    assert prep_rgba.mode == "RGB"

    # LA
    img_la = Image.new("LA", (100, 100), color=(200, 128))
    prep_la = prepare_image(img_la)
    assert prep_la.mode == "RGB"

    # Grayscale L
    img_l = Image.new("L", (100, 100), color=128)
    prep_l = prepare_image(img_l)
    assert prep_l.mode == "RGB"


def test_large_image_patch_inference(dummy_model_and_device):
    """2. Verifies patch inference on large 512x512 image."""
    model, device = dummy_model_and_device
    img = Image.new("RGB", (512, 512), color=(120, 60, 30))

    result = predict_image_patch_vote(img, model=model, device=device, n_patches=16, seed=42)

    assert result["label"] in ["FAKE", "REAL"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert 0.0 <= result["fake_probability"] <= 1.0
    assert 0.0 <= result["real_probability"] <= 1.0
    assert result["patch_count"] == 16
    assert result["inference_mode"] == "patch"
    assert len(result["patch_fake_probs"]) == 16
    assert len(result["patch_coordinates"]) == 16


def test_small_image_fallback(dummy_model_and_device):
    """3. Verifies small image (<32x32) fallback to resize mode."""
    model, device = dummy_model_and_device
    img = Image.new("RGB", (24, 24), color=(50, 100, 150))

    result = predict_image_patch_vote(img, model=model, device=device, n_patches=16)

    assert result["inference_mode"] == "resize_fallback"
    assert result["patch_count"] == 1


def test_patch_count_correctness(dummy_model_and_device):
    """4. Verifies requested patch count matches output patch count."""
    model, device = dummy_model_and_device
    img = Image.new("RGB", (300, 300), color=(10, 20, 30))

    res32 = predict_image_patch_vote(img, model=model, device=device, n_patches=32)
    assert res32["patch_count"] == 32

    res8 = predict_image_patch_vote(img, model=model, device=device, n_patches=8)
    assert res8["patch_count"] == 8


def test_deterministic_patch_coordinates():
    """5. Verifies same seed produces identical patch coordinates, different seeds differ."""
    img = Image.new("RGB", (400, 400), color=(200, 200, 200))

    _, coords1_a = extract_native_patches(img, patch_size=(32, 32), n_patches=32, seed=42)
    _, coords1_b = extract_native_patches(img, patch_size=(32, 32), n_patches=32, seed=42)
    _, coords2 = extract_native_patches(img, patch_size=(32, 32), n_patches=32, seed=99)

    assert coords1_a == coords1_b
    assert coords1_a != coords2


def test_patch_aggregation_methods(dummy_model_and_device):
    """6. Verifies all 4 aggregation strategies ('mean', 'median', 'majority', 'logit_mean')."""
    model, device = dummy_model_and_device
    img = Image.new("RGB", (256, 256), color=(80, 120, 160))

    for method in PATCH_AGGREGATION_METHODS:
        res = predict_image_patch_vote(img, model=model, device=device, n_patches=8, aggregation=method)
        assert res["aggregation"] == method
        assert 0.0 <= res["fake_probability"] <= 1.0
        assert 0.0 <= res["real_probability"] <= 1.0


def test_invalid_aggregation_handling(dummy_model_and_device):
    """7. Verifies ValueError on unsupported aggregation method."""
    model, device = dummy_model_and_device
    img = Image.new("RGB", (100, 100), color=(50, 50, 50))

    with pytest.raises(ValueError):
        predict_image_patch_vote(img, model=model, device=device, aggregation="invalid_agg_name")


def test_entropy_max_at_5050():
    """8. Verifies normalized entropy is 1.0 when probabilities are 50/50."""
    info = compute_prediction_entropy(0.5, 0.5)
    assert info["normalized_entropy"] == pytest.approx(1.0, abs=1e-3)
    assert "High Output Entropy" in info["uncertainty_level"]


def test_entropy_min_at_certain():
    """9. Verifies normalized entropy is near 0.0 for near-certain predictions."""
    info = compute_prediction_entropy(0.999, 0.001)
    assert info["normalized_entropy"] == pytest.approx(0.0, abs=0.05)
    assert "Low Output Entropy" in info["uncertainty_level"]


def test_entropy_range_validation():
    """10. Verifies normalized entropy is bounded in [0.0, 1.0] across range."""
    for p in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        info = compute_prediction_entropy(p, 1.0 - p)
        assert 0.0 <= info["normalized_entropy"] <= 1.0
        assert 0.0 <= info["entropy"] <= 1.0


def test_disagreement_analysis_metrics():
    """11. Verifies calculation of statistical disagreement metrics."""
    probs = [0.1, 0.2, 0.8, 0.9]
    dis = compute_prediction_disagreement(probs)

    assert dis["mean_fake_probability"] == pytest.approx(0.5, abs=1e-3)
    assert dis["median_fake_probability"] == pytest.approx(0.5, abs=1e-3)
    assert dis["min_fake_probability"] == pytest.approx(0.1, abs=1e-3)
    assert dis["max_fake_probability"] == pytest.approx(0.9, abs=1e-3)
    assert dis["range_fake_probability"] == pytest.approx(0.8, abs=1e-3)
    assert dis["fake_patch_count"] == 2
    assert dis["real_patch_count"] == 2
    assert dis["patch_agreement_pct"] == 50.0


def test_fft_spectral_diagnostic():
    """12. Verifies 2D FFT spectral diagnostic output structure and score bounds."""
    img = Image.new("RGB", (200, 200), color=(100, 100, 100))
    fft_res = compute_fft_spectral_diagnostic(img)

    assert "spectral_score" in fft_res
    assert "low_frequency_energy" in fft_res
    assert "high_frequency_energy" in fft_res
    assert "diagnostic_label" in fft_res
    assert "interpretation" in fft_res
    assert 0.0 <= fft_res["spectral_score"] <= 1.0


def test_hybrid_inference_output(dummy_model_and_device):
    """13. Verifies hybrid inference output structure and strategy agreement."""
    model, device = dummy_model_and_device
    img = Image.new("RGB", (256, 256), color=(40, 80, 120))

    result = predict_image_hybrid(img, model=model, device=device, n_patches=16)

    assert result["inference_mode"] == "hybrid"
    assert "resize_prediction" in result
    assert "patch_prediction" in result
    assert "prediction_difference" in result
    assert result["agreement"] in [
        "Strong Agreement", "Partial Agreement", "Moderate Agreement", "Strategy Disagreement",
        "Real Photo (Pattern Texture Filtered)",
        "Local AI Artifacts Detected", "Extreme Local AI Artifacts Detected",
        "Native Patch Confirmed (Aliasing Filtered)", "FFT Spectral Anomaly Detected",
    ]


def test_tta_inference_output(dummy_model_and_device):
    """14. Verifies test-time augmentation (TTA) inference structure."""
    model, device = dummy_model_and_device
    img = Image.new("RGB", (200, 200), color=(150, 150, 150))

    result = predict_image_tta(img, model=model, device=device)

    assert result["inference_mode"] == "tta"
    assert result["tta_view_count"] == 8
    assert len(result["tta_fake_probs"]) == 8
    assert 0.0 <= result["tta_std"] <= 1.0


def test_auto_mode_dispatch(dummy_model_and_device):
    """15. Verifies resolution-aware auto mode dispatcher."""
    model, device = dummy_model_and_device

    # Small image (32x32 < 64px threshold) -> auto selects resize
    img_small = Image.new("RGB", (32, 32), color=(10, 10, 10))
    res_small = predict_image_auto(img_small, model=model, device=device, mode="auto")
    assert res_small["inference_mode"] in ["resize", "resize_fallback"]

    # Medium image (100x100 >= 64px and < 256px threshold) -> auto selects patch
    img_medium = Image.new("RGB", (100, 100), color=(10, 10, 10))
    res_medium = predict_image_auto(img_medium, model=model, device=device, mode="auto")
    assert res_medium["inference_mode"] == "patch"

    # Large image (256x256 >= 256px threshold) -> auto selects hybrid
    img_large = Image.new("RGB", (256, 256), color=(10, 10, 10))
    res_large = predict_image_auto(img_large, model=model, device=device, mode="auto")
    assert res_large["inference_mode"] == "hybrid"


def test_backward_compatibility_predict_image(dummy_model_and_device):
    """16. Verifies original predict_image signature and behavior remain backward compatible."""
    model, device = dummy_model_and_device
    img = Image.new("RGB", (100, 100), color=(200, 100, 50))

    result = predict_image(img, model=model, device=device)

    assert "label" in result
    assert "confidence" in result
    assert "fake_probability" in result
    assert "real_probability" in result
    assert "confidence_info" in result
    assert result["inference_mode"] == "resize"
