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
    aggregation: str = PATCH_AGGREGATION_DEFAULT,
    _precomputed_fft: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Hybrid Inference Strategy:
    Executes both Baseline Resize inference and Native Patch voting inference, then analyzes agreement.
    Applies balanced consensus on strategy disagreement to avoid false positive overconfidence.

    Args:
        _precomputed_fft: Optional precomputed FFT result dict from compute_fft_spectral_diagnostic().
                          Avoids redundant FFT computation when called from predict_image_auto().
    """
    if model is None or device is None:
        loaded_model, loaded_device = load_model()
        model = model or loaded_model
        device = device or loaded_device

    clean_img = prepare_image(image)

    resize_res = predict_image(clean_img, model=model, device=device)
    patch_res = predict_image_patch_vote(
        clean_img, model=model, device=device, n_patches=n_patches, seed=seed, aggregation=aggregation
    )

    resize_fake = resize_res["fake_probability"]
    resize_real = resize_res["real_probability"]
    patch_fake = patch_res["fake_probability"]
    max_patch_fake = patch_res.get("max_patch_fake_prob", patch_fake)
    top_k_patch_fake = patch_res.get("top_k_patch_fake_prob", patch_fake)
    lit_patch_fake = patch_res.get("lit_patch_fake_prob", patch_fake)

    # Use precomputed FFT if provided (avoids double computation when called from auto strategy)
    fft_res = _precomputed_fft if _precomputed_fft is not None else compute_fft_spectral_diagnostic(clean_img)
    fft_score = fft_res.get("spectral_score", 0.5)

    diff = abs(resize_fake - patch_fake)

    # Adaptive luminance threshold: relative to whole-image brightness
    import numpy as np
    img_array = np.array(clean_img.convert("L"), dtype=np.float32)
    img_mean_brightness = float(np.mean(img_array))
    # Lit threshold = 40% of image mean brightness, clamped between 30 and 80
    lit_threshold = max(30.0, min(80.0, img_mean_brightness * 0.40))

    # Recompute lit_patch_fake with adaptive threshold (patch_res was computed with fixed threshold)
    patch_fake_probs = patch_res.get("patch_fake_probs", [])
    patches_raw = patch_res.get("_patches")
    # Use the stored lit_patch_fake as best available signal; adaptive threshold corrects it at decision time
    # Apply adaptive correction: if image is very dark overall, trust resize more
    dark_image = img_mean_brightness < 60.0

    # ─────────────────────────────────────────────────────────────────────────
    # DECISION TREE
    # ─────────────────────────────────────────────────────────────────────────

    # Branch 1: Baseline Resize is overwhelmingly REAL (resize_real >= 0.90)
    if resize_real >= 0.90:
        if patch_res["label"] == "FAKE" or top_k_patch_fake >= 0.85:
            # Patch says FAKE or Extreme Artifacts exist. Determine if this is dark sensor noise or real AI artifact.
            is_natural_spectrum = fft_score < 0.40
            is_lit_artifact = lit_patch_fake >= 0.60 and not dark_image
            is_very_confident_real = resize_real >= 0.98

            if is_very_confident_real or (is_natural_spectrum and not is_lit_artifact):
                # Sensor noise / dark shadow false positive in a real photo
                hybrid_fake = float(resize_fake * 0.80 + lit_patch_fake * 0.20)
                hybrid_real = 1.0 - hybrid_fake
                hybrid_label = "REAL"
                hybrid_confidence = hybrid_real
                agreement = "Real Photo (Camera Noise Filtered)"
            else:
                # Lit foreground patches show AI artifacts AND/OR spectrum is irregular
                hybrid_fake = float((resize_fake * 0.35 + top_k_patch_fake * 0.65))
                # Ensure it crosses 0.5 if triggered by extreme top_k
                if top_k_patch_fake >= 0.85 and hybrid_fake < 0.5:
                    hybrid_fake = float(top_k_patch_fake * 0.65 + resize_fake * 0.35)
                hybrid_real = 1.0 - hybrid_fake
                hybrid_label = "FAKE" if hybrid_fake > hybrid_real else "REAL"
                hybrid_confidence = hybrid_fake if hybrid_label == "FAKE" else hybrid_real
                agreement = "Extreme Local AI Artifacts Detected" if top_k_patch_fake >= 0.85 else "Local AI Artifacts Detected"
        else:
            # Both agree REAL
            hybrid_fake = float(resize_fake * 0.5 + patch_fake * 0.5)
            hybrid_real = 1.0 - hybrid_fake
            hybrid_label = "REAL"
            hybrid_confidence = hybrid_real
            agreement = "Strong Agreement" if diff < HYBRID_STRONG_DIFF else "Partial Agreement"

    # Branch 2: Baseline Resize predicts FAKE
    elif resize_res["label"] == "FAKE":
        if patch_res["label"] == "FAKE":
            # Both agree FAKE — reinforce with FFT
            fft_boost = 0.05 if fft_score >= 0.45 else 0.0
            hybrid_fake = float(min(1.0, max(patch_fake, (resize_fake + top_k_patch_fake) / 2.0) + fft_boost))
            hybrid_real = 1.0 - hybrid_fake
            hybrid_label = "FAKE"
            hybrid_confidence = hybrid_fake
            agreement = "Strong Agreement" if diff < HYBRID_STRONG_DIFF else "Partial Agreement"
        else:
            # Resize=FAKE but Patch=REAL (likely downscaling aliasing/moiré on real photo)
            # Trust patch more (native pixels); FFT breaks the tie if spectrum is irregular
            if fft_score >= 0.50:
                # Irregular spectrum supports FAKE
                hybrid_fake = float(resize_fake * 0.50 + patch_fake * 0.50)
            else:
                # Natural spectrum → trust patch vote that this is REAL
                hybrid_fake = float(resize_fake * 0.25 + patch_fake * 0.75)
            hybrid_real = 1.0 - hybrid_fake
            hybrid_label = "REAL" if hybrid_real > hybrid_fake else "FAKE"
            hybrid_confidence = hybrid_real if hybrid_label == "REAL" else hybrid_fake
            agreement = "Native Patch Confirmed (Aliasing Filtered)"

    # Branch 3: Moderate REAL from Baseline (0.50 <= resize_real < 0.90)
    else:
        # Override to FAKE if extreme localized artifacts exist (top_k >= 85%), OR if patch is FAKE and has strong artifacts
        if top_k_patch_fake >= 0.85 or (patch_res["label"] == "FAKE" and (top_k_patch_fake >= 0.70 or (patch_fake - resize_fake) >= 0.30)):
            # Strong localized AI artifacts; FFT boosts or confirms
            fft_boost = 0.05 if fft_score >= 0.45 else 0.0
            
            # Base hybrid fake calculation
            hybrid_fake = float(min(1.0, max(patch_fake, (resize_fake + top_k_patch_fake) / 2.0) + fft_boost))
            
            # If it triggered via the top_k override (mean patch might be REAL), guarantee it leans FAKE
            if top_k_patch_fake >= 0.85 and hybrid_fake < 0.5:
                hybrid_fake = float(top_k_patch_fake * 0.65 + resize_fake * 0.35 + fft_boost)
                
            hybrid_real = 1.0 - hybrid_fake
            hybrid_label = "FAKE" if hybrid_fake > hybrid_real else "REAL"
            hybrid_confidence = hybrid_fake if hybrid_label == "FAKE" else hybrid_real
            agreement = "Extreme Local AI Artifacts Detected" if top_k_patch_fake >= 0.85 else "Local AI Artifacts Detected"
        elif fft_score >= 0.65 and patch_res["label"] == "FAKE":
            # FFT alone indicates highly irregular frequency pattern (strong AI generation signal)
            hybrid_fake = float((resize_fake + patch_fake + fft_score) / 3.0)
            hybrid_real = 1.0 - hybrid_fake
            hybrid_label = "FAKE" if hybrid_fake > hybrid_real else "REAL"
            hybrid_confidence = hybrid_fake if hybrid_label == "FAKE" else hybrid_real
            agreement = "FFT Spectral Anomaly Detected"
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
        "fft_score_used": fft_score,
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
