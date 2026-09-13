import logging
from typing import Dict, Any, List
from PIL import Image
import torch
import pandas as pd

from app.config import (
    DEFAULT_INFERENCE_MODE,
    INFERENCE_MODES,
    PATCH_AGGREGATION_DEFAULT,
)
from app.model_loader import load_model
from app.diagnostics.entropy import compute_prediction_entropy
from app.diagnostics.disagreement import compute_prediction_disagreement
from app.diagnostics.fft_spectral import compute_fft_spectral_diagnostic
from app.strategies.patch.patch_extractor import prepare_image
from app.strategies.resize.resize_strategy import predict_image
from app.strategies.patch.patch_strategy import predict_image_patch_vote
from app.strategies.tta.tta_strategy import predict_image_tta
from app.strategies.hybrid.hybrid_strategy import predict_image_hybrid


def predict_image_auto(
    image: Image.Image,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    mode: str = DEFAULT_INFERENCE_MODE,
    n_patches: int = 0,  # 0 enables dynamic patch count based on resolution
    seed: int = 42,
    aggregation: str = PATCH_AGGREGATION_DEFAULT
) -> Dict[str, Any]:
    """
    Unified automatic dispatcher supporting modes: 'auto', 'resize', 'patch', 'hybrid', 'tta'.
    """
    if mode not in INFERENCE_MODES:
        raise ValueError(f"Invalid inference mode '{mode}'. Supported: {INFERENCE_MODES}")

    clean_img = prepare_image(image)
    w, h = clean_img.size
    min_dim = min(w, h)

    # Resolution-aware automatic selection logic
    selected_mode = mode
    if mode == "auto":
        if min_dim < 64:
            selected_mode = "resize"
        elif min_dim < 256:
            selected_mode = "patch"
        else:
            selected_mode = "hybrid"

    # Compute FFT once at the dispatcher level for hybrid/auto to avoid double computation
    fft_diagnostic = compute_fft_spectral_diagnostic(clean_img)

    if selected_mode == "resize":
        result = predict_image(clean_img, model=model, device=device)
    elif selected_mode == "patch":
        result = predict_image_patch_vote(
            clean_img, model=model, device=device, n_patches=n_patches, seed=seed, aggregation=aggregation
        )
    elif selected_mode == "hybrid":
        result = predict_image_hybrid(
            clean_img, model=model, device=device, n_patches=n_patches, seed=seed, aggregation=aggregation,
            _precomputed_fft=fft_diagnostic
        )
    elif selected_mode == "tta":
        result = predict_image_tta(clean_img, model=model, device=device)

    # Attach entropy, disagreement, and FFT diagnostics to the output dictionary
    entropy_info = compute_prediction_entropy(result["fake_probability"], result["real_probability"])
    result.update(entropy_info)

    if "patch_fake_probs" in result:
        disagreement_info = compute_prediction_disagreement(result["patch_fake_probs"])
        result["stability"] = disagreement_info

    result["fft_diagnostic"] = fft_diagnostic
    result["image_dimensions"] = f"{w} x {h}"
    return result


def predict_batch(
    images_dict: Dict[str, Image.Image],
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    mode: str = DEFAULT_INFERENCE_MODE,
    n_patches: int = 0,
    aggregation: str = PATCH_AGGREGATION_DEFAULT
) -> pd.DataFrame:
    """
    Processes a dictionary of {filename: PIL.Image} and returns a structured pandas DataFrame.
    """
    if model is None or device is None:
        loaded_model, loaded_device = load_model()
        model = model or loaded_model
        device = device or loaded_device

    results: List[Dict[str, Any]] = []

    for idx, (filename, img) in enumerate(images_dict.items()):
        try:
            res = predict_image_auto(
                img, model=model, device=device, mode=mode, n_patches=n_patches, aggregation=aggregation
            )
            results.append({
                "Filename": filename,
                "Prediction": res["label"],
                "Confidence": f"{res['confidence'] * 100:.2f}%",
                "Fake Probability": f"{res['fake_probability'] * 100:.2f}%",
                "Real Probability": f"{res['real_probability'] * 100:.2f}%",
                "Inference Mode": res.get("inference_mode", mode),
                "Uncertainty": res.get("uncertainty_level", "N/A"),
                "Raw Confidence": res["confidence"]
            })
        except Exception as e:
            logging.error(f"Error processing {filename}: {e}", exc_info=True)
            results.append({
                "Filename": filename,
                "Prediction": "ERROR",
                "Confidence": "N/A",
                "Fake Probability": "N/A",
                "Real Probability": "N/A",
                "Inference Mode": mode,
                "Uncertainty": "ERROR",
                "Raw Confidence": 0.0
            })

    return pd.DataFrame(results)
