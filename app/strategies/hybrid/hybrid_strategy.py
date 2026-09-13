from typing import Dict, Any
from PIL import Image
import torch

from app.config import (
    PATCH_N,
    PATCH_AGGREGATION_DEFAULT,
    HYBRID_STRONG_DIFF,
    HYBRID_PARTIAL_DIFF,
)
from app.model_loader import load_model
from app.diagnostics.entropy import interpret_confidence
from app.diagnostics.fft_spectral import compute_fft_spectral_diagnostic
from app.strategies.patch.patch_extractor import prepare_image
from app.strategies.resize.resize_strategy import predict_image
from app.strategies.patch.patch_strategy import predict_image_patch_vote


def predict_image_hybrid(
    image: Image.Image,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    n_patches: int = PATCH_N,
    seed: int = 42,
    aggregation: str = PATCH_AGGREGATION_DEFAULT
) -> Dict[str, Any]:
    """
    Hybrid Inference Strategy:
    Executes both Baseline Resize inference and Native Patch voting inference, then analyzes agreement.
    Applies balanced consensus on strategy disagreement to avoid false positive overconfidence.
    """
    if model is None or device is None:
        loaded_model, loaded_device = load_model()
        model = model or loaded_model
        device = device or loaded_device

    resize_res = predict_image(image, model=model, device=device)
    patch_res = predict_image_patch_vote(
        image, model=model, device=device, n_patches=n_patches, seed=seed, aggregation=aggregation
    )

    resize_fake = resize_res["fake_probability"]
    resize_real = resize_res["real_probability"]
    patch_fake = patch_res["fake_probability"]
    max_patch_fake = patch_res.get("max_patch_fake_prob", patch_fake)
    top_k_patch_fake = patch_res.get("top_k_patch_fake_prob", patch_fake)
    lit_patch_fake = patch_res.get("lit_patch_fake_prob", patch_fake)

    clean_img = prepare_image(image)
    fft_res = compute_fft_spectral_diagnostic(clean_img)

    diff = abs(resize_fake - patch_fake)

    # 1. Baseline Resize is Overwhelmingly REAL (resize_real >= 0.90):
    if resize_real >= 0.90:
        if patch_res["label"] == "FAKE":
            # Native patch voting says FAKE. Is this dark camera sensor noise / shadow grain?
            # Or genuine high-res AI image with localized AI artifacts on lit foreground patches?
            if lit_patch_fake < 0.60 or resize_real >= 0.98:
                # Real photograph with dark shadow / camera sensor noise.
                # Baseline 32x32 resize correctly identified the real photo semantics (>=90-98% REAL).
                hybrid_fake = float(resize_fake * 0.80 + lit_patch_fake * 0.20)
                hybrid_real = 1.0 - hybrid_fake
                hybrid_label = "REAL"
                hybrid_confidence = hybrid_real
                agreement = "Real Photo (Camera Noise Filtered)"
            else:
                # Lit/foreground patches ALSO show strong AI artifacts (high-res Gemini/DALL-E portrait)
                hybrid_fake = float((resize_fake + top_k_patch_fake) / 2.0)
                hybrid_real = 1.0 - hybrid_fake
                hybrid_label = "FAKE" if hybrid_fake > hybrid_real else "REAL"
                hybrid_confidence = hybrid_fake if hybrid_label == "FAKE" else hybrid_real
                agreement = "Local AI Artifacts Detected"
        else:
            # Both Baseline Resize and Patch Voting agree REAL!
            hybrid_fake = float(resize_fake * 0.5 + patch_fake * 0.5)
            hybrid_real = 1.0 - hybrid_fake
            hybrid_label = "REAL"
            hybrid_confidence = hybrid_real
            agreement = "Strong Agreement" if diff < HYBRID_STRONG_DIFF else "Partial Agreement"

    # 2. Baseline Resize predicts FAKE (resize_fake >= 0.50):
    elif resize_res["label"] == "FAKE":
        if patch_res["label"] == "FAKE":
            # Both agree FAKE!
            hybrid_fake = float(max(patch_fake, (resize_fake + top_k_patch_fake) / 2.0))
            hybrid_real = 1.0 - hybrid_fake
            hybrid_label = "FAKE"
            hybrid_confidence = hybrid_fake
            agreement = "Strong Agreement" if diff < HYBRID_STRONG_DIFF else "Partial Agreement"
        else:
            # Baseline Resize said FAKE (due to downscaling aliasing/moire), but Patch Voting said REAL!
            hybrid_fake = float(resize_fake * 0.30 + patch_fake * 0.70)
            hybrid_real = 1.0 - hybrid_fake
            hybrid_label = "REAL" if hybrid_real > hybrid_fake else "FAKE"
            hybrid_confidence = hybrid_real if hybrid_label == "REAL" else hybrid_fake
            agreement = "Native Patch Confirmed (Aliasing Filtered)"

    # 3. Baseline Resize is Moderate REAL (0.50 <= resize_real < 0.90):
    else:
        if patch_res["label"] == "FAKE" and (top_k_patch_fake >= 0.75 or (patch_fake - resize_fake) >= 0.35):
            # Localized AI artifacts in high-res AI image (e.g. Gemini / Midjourney)
            hybrid_fake = float(max(patch_fake, (resize_fake + top_k_patch_fake) / 2.0))
            hybrid_real = 1.0 - hybrid_fake
            hybrid_label = "FAKE" if hybrid_fake > hybrid_real else "REAL"
            hybrid_confidence = hybrid_fake if hybrid_label == "FAKE" else hybrid_real
            agreement = "Local AI Artifacts Detected"
        else:
            hybrid_fake = float((resize_fake + patch_fake) / 2.0)
            hybrid_real = 1.0 - hybrid_fake
            hybrid_label = "FAKE" if hybrid_fake > hybrid_real else "REAL"
            hybrid_confidence = hybrid_fake if hybrid_label == "FAKE" else hybrid_real
            agreement = "Strategy Disagreement" if resize_res["label"] != patch_res["label"] else "Moderate Agreement"

    top_k_label = "FAKE" if top_k_patch_fake >= 0.5 else "REAL"

    return {
        "label": hybrid_label,
        "confidence": hybrid_confidence,
        "fake_probability": hybrid_fake,
        "real_probability": hybrid_real,
        "prediction_difference": float(diff),
        "agreement": agreement,
        "resize_prediction": {
            "label": resize_res["label"],
            "fake_probability": resize_res["fake_probability"],
            "real_probability": resize_res["real_probability"]
        },
        "patch_prediction": {
            "label": patch_res["label"],
            "fake_probability": patch_res["fake_probability"],
            "real_probability": patch_res["real_probability"],
            "patch_count": patch_res["patch_count"],
            "patch_fake_probs": patch_res.get("patch_fake_probs", []),
            "max_patch_fake_prob": max_patch_fake,
            "top_k_patch_fake_prob": top_k_patch_fake
        },
        "top_k_patch_prediction": {
            "label": top_k_label,
            "fake_probability": top_k_patch_fake,
            "real_probability": 1.0 - top_k_patch_fake
        },
        "fft_diagnostic": fft_res,
        "inference_mode": "hybrid",
        "confidence_info": interpret_confidence(hybrid_confidence)
    }
