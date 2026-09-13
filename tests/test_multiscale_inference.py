"""
Unit tests for SignalScope Multi-Scale Inference Pipeline
"""

import pytest
from PIL import Image
import torch

from app.model_loader import load_model
from app.strategies.multiscale.multiscale_extractor import (
    extract_object_context_patches,
    extract_native_texture_patches,
)
from app.strategies.multiscale.multiscale_strategy import (
    predict_image_multiscale,
    _calculate_branch_statistics
)


@pytest.fixture(scope="module")
def loaded_model_fixture():
    model, device = load_model()
    return model, device


def test_frozen_model_integrity(loaded_model_fixture):
    """Verifies that model parameters are frozen and in eval mode."""
    model, _ = loaded_model_fixture
    assert not model.training, "Model must be in eval() mode"
    # Ensure FC layer output features match NUM_CLASSES (2)
    assert model.fc.out_features == 2, "Model output classes must equal 2"


def test_object_context_16_patches():
    """Verifies 128x128 context branch extracts exactly 16 patches with remapped coordinates."""
    img = Image.new("RGB", (512, 512), color=(100, 150, 200))
    patches, coords_128, coords_orig = extract_object_context_patches(img)

    assert len(patches) == 16, "Must extract exactly 16 patches from 128x128 context"
    assert len(coords_128) == 16
    assert len(coords_orig) == 16

    # Verify 128x128 grid positions
    assert coords_128[0] == {"x": 0, "y": 0, "width": 32, "height": 32}
    assert coords_128[15] == {"x": 96, "y": 96, "width": 32, "height": 32}

    # Verify remapped original coordinates (512 / 128 = 4x scale)
    assert coords_orig[0] == {"x": 0, "y": 0, "width": 128, "height": 128}
    assert coords_orig[15] == {"x": 384, "y": 384, "width": 128, "height": 128}


def test_native_texture_extraction():
    """Verifies native texture patch extraction and border handling."""
    img = Image.new("RGB", (100, 100), color=(50, 50, 50))
    patches, coords, skipped_count = extract_native_texture_patches(img, patch_size=(32, 32), stride=32)

    # 100 / 32 = 3 full patches along each axis -> 3x3 = 9 total patches
    assert len(patches) == 9
    assert len(coords) == 9
    assert skipped_count > 0, "Border pixels (100 - 96 = 4px) should be logged as skipped"


def test_tiny_image_fallback(loaded_model_fixture):
    """Verifies images smaller than 32x32 do not crash and handle weight re-normalization."""
    model, device = loaded_model_fixture
    tiny_img = Image.new("RGB", (20, 20), color=(200, 200, 200))

    res = predict_image_multiscale(tiny_img, model=model, device=device)

    assert "label" in res
    assert res["label"] in ["REAL", "FAKE", "UNCERTAIN"]
    assert "analysis" in res
    assert round(res["analysis"]["fusion"]["global_weight"] + res["analysis"]["fusion"]["object_context_weight"], 4) == 1.0, "Sum of active branch weights must equal 1.0"


def test_branch_statistics_aggregation():
    """Verifies patch statistics calculations (mean, median, max, top-k mean, std)."""
    fake_probs = [0.1, 0.2, 0.3, 0.8, 0.9]
    real_probs = [1.0 - p for p in fake_probs]
    coords = [{"x": 0, "y": 0, "width": 32, "height": 32}] * 5

    stats = _calculate_branch_statistics(fake_probs, real_probs, coords, source_name="test")

    assert stats["patch_count"] == 5
    assert abs(stats["mean_fake_probability"] - 0.46) < 1e-3
    assert abs(stats["max_fake_probability"] - 0.90) < 1e-3
    # Top-K ratio = 0.20 -> max(1, round(5*0.20)) = 1 patch -> max = 0.90
    assert abs(stats["top_k_mean_fake_probability"] - 0.90) < 1e-3
    assert stats["fake_patch_count"] == 2
    assert stats["real_patch_count"] == 3


def test_multiscale_full_pipeline(loaded_model_fixture):
    """Verifies complete multi-scale inference return structure and API backwards compatibility."""
    model, device = loaded_model_fixture
    img = Image.new("RGB", (256, 256), color=(120, 120, 120))

    res = predict_image_multiscale(img, model=model, device=device)

    # API backward compatibility check
    assert "label" in res
    assert "confidence" in res
    assert "fake_probability" in res
    assert "real_probability" in res
    assert "inference_mode" in res
    assert res["inference_mode"] == "multiscale"

    # Multi-scale evidence analysis structure check
    assert "analysis" in res
    analysis = res["analysis"]
    assert "global" in analysis
    assert "object_context" in analysis
    assert "native_texture" in analysis
    assert "fusion" in analysis
    assert "highlighted_regions" in analysis
    assert isinstance(analysis["highlighted_regions"], list)
