from typing import Dict, Any
from PIL import Image
import torch

from app.config import (
    PATCH_N,
    PATCH_AGGREGATION_DEFAULT,
    HYBRID_STRONG_DIFF,
    MULTISCALE_FAKE_THRESHOLD,
)
from app.model_loader import load_model, resolve_model_device
from app.diagnostics.entropy import interpret_confidence
from app.diagnostics.fft_spectral import compute_fft_spectral_diagnostic
from app.strategies.patch.patch_extractor import prepare_image
from app.strategies.resize.resize_strategy import predict_image
from app.strategies.patch.patch_strategy import predict_image_patch_vote
from app.strategies.base_strategy import BaseStrategy, validate_strategy_output
from app.strategies.strategy_registry import register_strategy


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
    model, device = resolve_model_device(model, device)

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

    import numpy as np
    img_array = np.array(clean_img.convert("L"), dtype=np.float32)
    img_mean_brightness = float(np.mean(img_array))
    
    # Apply adaptive correction: if image is very dark overall, trust resize more
    dark_image = img_mean_brightness < 60.0

    # ─────────────────────────────────────────────────────────────────────────
    # DECISION TREE
    # ─────────────────────────────────────────────────────────────────────────

    # Rule 0: Extreme Localized AI Artifact Override
    # High-res AI images (Gemini, Midjourney) often have smooth backgrounds (sky/walls) 
    # that fool baseline resize, but contain severe AI artifacts in key patches.
    # ONLY trigger if resize is NOT overwhelmingly REAL (< 0.90) AND patch vote is strongly fake (>= 0.55).
    if top_k_patch_fake >= 0.92 and resize_fake >= 0.40 and resize_real < 0.90 and patch_fake >= 0.55 and fft_score >= 0.45 and not (dark_image and lit_patch_fake < 0.40):
        hybrid_fake = float(top_k_patch_fake * 0.70 + resize_fake * 0.30)
        hybrid_real = 1.0 - hybrid_fake
        hybrid_label = "FAKE"
        hybrid_confidence = hybrid_fake
        agreement = "Extreme Local AI Artifacts Detected"

    # Branch 1: Baseline Resize is overwhelmingly REAL (resize_real >= 0.90)
    elif resize_real >= 0.90:
        if resize_fake >= 0.40 and patch_fake >= 0.65 and top_k_patch_fake >= 0.92 and fft_score >= 0.50:
            # Strong evidence from multiple patches AND spectral irregularity required to override a >90% REAL baseline
            hybrid_fake = float((resize_fake * 0.35 + top_k_patch_fake * 0.65))
            hybrid_real = 1.0 - hybrid_fake
            hybrid_label = "FAKE" if hybrid_fake >= MULTISCALE_FAKE_THRESHOLD else "REAL"
            hybrid_confidence = hybrid_fake if hybrid_label == "FAKE" else hybrid_real
            agreement = "Extreme Local AI Artifacts Detected" if top_k_patch_fake >= 0.92 else "Local AI Artifacts Detected"
        else:
            # Baseline resize is overwhelmingly REAL (>= 90%) and patch artifacts are either weak or spectrally normal
            # High-frequency pattern texture (woven fabric/bedsheets/tablecloth/noise) in a real photo
            hybrid_fake = float(resize_fake * 0.75 + patch_fake * 0.25)
            hybrid_real = 1.0 - hybrid_fake
            hybrid_label = "REAL"
            hybrid_confidence = hybrid_real
            agreement = "Real Photo (Pattern Texture Filtered)"


    # Branch 2: Baseline Resize predicts FAKE (often distorted by 32x32 downscaling aliasing on high-res camera photos)
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
            # Resize=FAKE but Native Patch Voting=REAL (32x32 downscaling aliasing on high-res photo)
            # Trust native 1:1 pixel patches (patch_fake) which are free of downscaling distortion
            hybrid_fake = float(patch_fake * 0.80 + resize_fake * 0.20)
            hybrid_real = 1.0 - hybrid_fake
            hybrid_label = "FAKE" if hybrid_fake >= MULTISCALE_FAKE_THRESHOLD else "REAL"
            hybrid_confidence = hybrid_real if hybrid_label == "REAL" else hybrid_fake
            agreement = "Native Patch Confirmed (Aliasing Filtered)"


    # Branch 3: Moderate REAL from Baseline (0.50 <= resize_real < 0.90)
    else:
        # Override to FAKE if extreme localized artifacts exist (top_k >= 85%), OR if patch is FAKE and has strong artifacts
        if resize_fake >= 0.40 and ((top_k_patch_fake >= 0.90 and fft_score >= 0.45) or (patch_res["label"] == "FAKE" and (top_k_patch_fake >= 0.75 or (patch_fake - resize_fake) >= 0.35))):
            # Strong localized AI artifacts; FFT boosts or confirms
            fft_boost = 0.05 if fft_score >= 0.45 else 0.0
            
            # Base hybrid fake calculation
            hybrid_fake = float(min(1.0, max(patch_fake, (resize_fake + top_k_patch_fake) / 2.0) + fft_boost))
            
            # If it triggered via the top_k override (mean patch might be REAL), guarantee it leans FAKE
            if top_k_patch_fake >= 0.90 and hybrid_fake < 0.5:
                hybrid_fake = float(top_k_patch_fake * 0.65 + resize_fake * 0.35 + fft_boost)
                
            hybrid_real = 1.0 - hybrid_fake
            hybrid_label = "FAKE" if hybrid_fake >= MULTISCALE_FAKE_THRESHOLD else "REAL"
            hybrid_confidence = hybrid_fake if hybrid_label == "FAKE" else hybrid_real
            agreement = "Extreme Local AI Artifacts Detected" if top_k_patch_fake >= 0.90 else "Local AI Artifacts Detected"
        elif fft_score >= 0.65 and patch_res["label"] == "FAKE":
            # FFT alone indicates highly irregular frequency pattern (strong AI generation signal)
            hybrid_fake = float((resize_fake + patch_fake + fft_score) / 3.0)
            hybrid_real = 1.0 - hybrid_fake
            hybrid_label = "FAKE" if hybrid_fake >= MULTISCALE_FAKE_THRESHOLD else "REAL"
            hybrid_confidence = hybrid_fake if hybrid_label == "FAKE" else hybrid_real
            agreement = "FFT Spectral Anomaly Detected"
        else:
            hybrid_fake = float((resize_fake + patch_fake) / 2.0)
            hybrid_real = 1.0 - hybrid_fake
            hybrid_label = "FAKE" if hybrid_fake >= MULTISCALE_FAKE_THRESHOLD else "REAL"
            hybrid_confidence = hybrid_fake if hybrid_label == "FAKE" else hybrid_real
            agreement = "Strategy Disagreement" if resize_res["label"] != patch_res["label"] else "Moderate Agreement"

    top_k_label = "FAKE" if top_k_patch_fake >= 0.5 else "REAL"

    res = {
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
        "threshold": MULTISCALE_FAKE_THRESHOLD,
        "confidence_info": interpret_confidence(hybrid_confidence)
    }
    return validate_strategy_output(res, strategy_name="hybrid")




class HybridStrategy(BaseStrategy):
    """Concrete BaseStrategy implementation for Hybrid Fusion Strategy."""

    @property
    def name(self) -> str:
        return "hybrid"

    @property
    def display_name(self) -> str:
        return "Hybrid Fusion (Resize + Patch + FFT)"

    @property
    def description(self) -> str:
        return "Combines single-pass resize, native patch voting, and FFT spectral anomaly diagnostics."

    def predict(
        self,
        image: Image.Image,
        model: torch.nn.Module | None = None,
        device: torch.device | None = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        return predict_image_hybrid(image, model=model, device=device, **kwargs)


# Register strategy with StrategyRegistry
register_strategy(HybridStrategy())