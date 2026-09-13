"""
Unit tests for SignalScope Strategy Registry and BaseStrategy implementations.
"""

import pytest
from PIL import Image
import torch

from app.strategies.base_strategy import BaseStrategy, validate_strategy_output
from app.strategies.strategy_registry import StrategyRegistry, get_strategy, list_strategies
from app.model_loader import load_model


@pytest.fixture(scope="module")
def model_and_device():
    model, device = load_model()
    return model, device


@pytest.fixture
def sample_image():
    # 256x256 RGB image for strategy testing
    return Image.new("RGB", (256, 256), color=(128, 128, 128))


def test_strategy_registry_listing():
    strategies = list_strategies()
    keys = [s["key"] for s in strategies]
    assert "multiscale" in keys
    assert "auto" in keys
    assert "patch" in keys
    assert "hybrid" in keys
    assert "resize" in keys
    assert "tta" in keys


def test_base_strategy_output_validation():
    valid_res = {
        "label": "REAL",
        "confidence": 0.95,
        "fake_probability": 0.05,
        "real_probability": 0.95,
        "inference_mode": "test_mode"
    }
    validated = validate_strategy_output(valid_res, strategy_name="test")
    assert validated["label"] == "REAL"
    assert validated["fake_probability"] == 0.05

    invalid_res = {
        "confidence": 0.95,
        "fake_probability": 0.05,
    }
    with pytest.raises(ValueError, match="missing required keys"):
        validate_strategy_output(invalid_res, strategy_name="test_invalid")


def test_run_multiscale_via_registry(model_and_device, sample_image):
    model, device = model_and_device
    strat = get_strategy("multiscale")
    assert strat.name == "multiscale"

    res = StrategyRegistry.run_strategy("multiscale", sample_image, model=model, device=device)
    assert res["label"] in ("REAL", "FAKE", "UNCERTAIN")
    assert "fake_probability" in res
    assert "real_probability" in res
    assert res["inference_mode"] == "multiscale"


def test_run_auto_via_registry(model_and_device, sample_image):
    model, device = model_and_device
    res = StrategyRegistry.run_strategy("auto", sample_image, model=model, device=device)
    assert res["label"] in ("REAL", "FAKE", "UNCERTAIN")
    assert "confidence" in res


def test_run_patch_via_registry(model_and_device, sample_image):
    model, device = model_and_device
    res = StrategyRegistry.run_strategy("patch", sample_image, model=model, device=device)
    assert res["label"] in ("REAL", "FAKE")
    assert res["inference_mode"] == "patch"


def test_run_resize_via_registry(model_and_device, sample_image):
    model, device = model_and_device
    res = StrategyRegistry.run_strategy("resize", sample_image, model=model, device=device)
    assert res["label"] in ("REAL", "FAKE")
    assert res["inference_mode"] == "resize"
