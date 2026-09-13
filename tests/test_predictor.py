"""
Unit Tests for SignalScope Model Loader and Predictor Engine
"""

from pathlib import Path
import pytest
from PIL import Image
import torch

from app.config import MODEL_PATH, IMAGE_SIZE
from app.model_loader import load_model, get_device
from app.predictor import preprocess_image, predict_image, predict_batch


@pytest.fixture(scope="module")
def loaded_model_fixture():
    """
    Module-scoped fixture to load model once for tests.
    """
    assert MODEL_PATH.exists(), f"Model file missing at: {MODEL_PATH}"
    model, device = load_model(MODEL_PATH)
    return model, device


def test_get_device():
    """
    Verifies that get_device returns a valid torch device instance.
    """
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ["cuda", "cpu"]


def test_model_loading(loaded_model_fixture):
    """
    Verifies model architecture initialization and eval mode.
    """
    model, device = loaded_model_fixture
    assert model is not None
    assert isinstance(model, torch.nn.Module)
    assert not model.training  # Must be in eval() mode


def test_preprocess_rgb_image():
    """
    Verifies preprocessing output shape and normalization for RGB PIL Image.
    """
    img = Image.new("RGB", (300, 300), color=(128, 64, 32))
    tensor = preprocess_image(img)
    assert tensor.shape == (1, 3, *IMAGE_SIZE)
    assert isinstance(tensor, torch.Tensor)


def test_preprocess_grayscale_image():
    """
    Verifies grayscale (Mode 'L') image auto-conversion to 3-channel tensor.
    """
    img = Image.new("L", (150, 150), color=200)
    tensor = preprocess_image(img)
    assert tensor.shape == (1, 3, *IMAGE_SIZE)


def test_preprocess_rgba_image():
    """
    Verifies RGBA image auto-conversion to 3-channel RGB tensor.
    """
    img = Image.new("RGBA", (200, 200), color=(255, 0, 0, 128))
    tensor = preprocess_image(img)
    assert tensor.shape == (1, 3, *IMAGE_SIZE)


def test_predict_image_output_structure(loaded_model_fixture):
    """
    Verifies single image prediction dictionary keys, data types, and probability constraints.
    """
    model, device = loaded_model_fixture
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    result = predict_image(img, model=model, device=device)

    # Required Output Keys
    assert "label" in result
    assert "confidence" in result
    assert "fake_probability" in result
    assert "real_probability" in result
    assert "confidence_info" in result

    # Label constraints
    assert result["label"] in ["FAKE", "REAL"]

    # Probability bounds
    fake_p = result["fake_probability"]
    real_p = result["real_probability"]
    conf = result["confidence"]

    assert 0.0 <= fake_p <= 1.0
    assert 0.0 <= real_p <= 1.0
    assert 0.0 <= conf <= 1.0

    # Sum of probabilities approximately 1.0
    assert (fake_p + real_p) == pytest.approx(1.0, abs=1e-4)

    # Confidence matches the predicted label probability
    if result["label"] == "FAKE":
        assert conf == pytest.approx(fake_p)
    else:
        assert conf == pytest.approx(real_p)


def test_predict_batch(loaded_model_fixture):
    """
    Verifies batch prediction pandas DataFrame format.
    """
    model, device = loaded_model_fixture
    images_dict = {
        "sample1.png": Image.new("RGB", (224, 224), color="blue"),
        "sample2.jpg": Image.new("RGB", (100, 100), color="red")
    }

    df = predict_batch(images_dict, model=model, device=device)
    assert len(df) == 2
    assert "Filename" in df.columns
    assert "Prediction" in df.columns
    assert "Confidence" in df.columns
    assert "Fake Probability" in df.columns
    assert "Real Probability" in df.columns


def test_invalid_input_handling():
    """
    Verifies proper error handling when input is not a PIL Image.
    """
    with pytest.raises(ValueError):
        preprocess_image("not_an_image")
