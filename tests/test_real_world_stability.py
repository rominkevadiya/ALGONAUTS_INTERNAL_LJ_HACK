"""
Unit tests for checking real-world stability of the inference pipeline.
"""
import pytest
import numpy as np
import urllib.request
from PIL import Image
from io import BytesIO
import torch
import warnings

from app.predictor import predict_image_hybrid
from app.model_loader import load_model

def get_synthetic_texture():
    """Generates a synthetic natural texture image (e.g. base color + noise)."""
    h, w = 512, 512
    # Base color (green-ish/brown-ish)
    base = np.random.randint(100, 180, (h, w, 3), dtype=np.int16)
    # Gaussian-like noise
    noise = np.random.randint(-30, 31, (h, w, 3), dtype=np.int16)
    img_arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img_arr)


@pytest.fixture(scope="module")
def real_model_and_device():
    """Load the actual trained model for stability tests."""
    model, device = load_model()
    return model, device


def test_stability_across_patch_counts(real_model_and_device):
    """
    Asserts that for a synthetic natural texture image, the hybrid prediction
    label remains STABLE across different n_patches, eliminating non-deterministic flipping.
    """
    model, device = real_model_and_device
    img = get_synthetic_texture()
    
    labels = []
    for n_patches in [16, 24, 32, 48, 64]:
        res = predict_image_hybrid(img, model=model, device=device, n_patches=n_patches, seed=42)
        labels.append(res["label"])
        
    # Assert all labels are the same
    assert len(set(labels)) == 1, f"Instability detected across patch counts! Labels: {labels}"


def test_stability_across_seeds(real_model_and_device):
    """
    Asserts that the hybrid prediction label is STABLE across different seeds
    for patch extraction on a noisy texture image.
    """
    model, device = real_model_and_device
    img = get_synthetic_texture()
    
    labels = []
    for seed in [1, 5, 10, 42, 100]:
        res = predict_image_hybrid(img, model=model, device=device, n_patches=32, seed=seed)
        labels.append(res["label"])
        
    # Assert all labels are the same
    assert len(set(labels)) == 1, f"Instability detected across seeds! Labels: {labels}"


def fetch_image(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            return Image.open(BytesIO(response.read())).convert("RGB")
    except Exception as e:
        warnings.warn(f"Failed to fetch {url}: {e}")
        return None

def test_real_photos_not_falsely_flagged(real_model_and_device):
    """
    Fetches real public-domain images and asserts they are NOT classified
    as FAKE at >70% confidence. Skips gracefully if network is unavailable.
    """
    model, device = real_model_and_device
    
    urls = [
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg",
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/butterfly.jpg",
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
    ]
    
    for url in urls:
        img = fetch_image(url)
        if img is None:
            pytest.skip("Network unavailable or fetch failed, skipping real image test.")
            
        res = predict_image_hybrid(img, model=model, device=device)
        
        # We assert that the fake probability is NOT > 0.70.
        # This prevents extreme false positive confidence on known real images.
        assert res["fake_probability"] <= 0.70, (
            f"Image {url} falsely flagged with high confidence FAKE: {res['fake_probability']:.2%}"
        )
