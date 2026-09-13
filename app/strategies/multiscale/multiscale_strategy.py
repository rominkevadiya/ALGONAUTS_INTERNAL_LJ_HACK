"""
SignalScope Multi-Scale Inference & Evidence-Generation Strategy
Wraps existing frozen ResNet-50 model with Global (32x32), Object/Context (128x128 -> 16 patches),
and Native Texture (32x32 crops) analysis branches.
Combines evidence using transparent weighted fusion and tri-state decision logic (REAL, FAKE, UNCERTAIN).
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.config import (
    CLASS_MAPPING,
    MULTISCALE_GLOBAL_WEIGHT,
    MULTISCALE_CONTEXT_WEIGHT,
    MULTISCALE_TEXTURE_WEIGHT,
    MULTISCALE_TOP_K_RATIO,
    MULTISCALE_FAKE_THRESHOLD,
    MULTISCALE_REAL_THRESHOLD,
    INFERENCE_BATCH_SIZE,
    MAX_NATIVE_PATCHES,
)
from app.model_loader import load_model
from app.diagnostics.entropy import interpret_confidence
from app.strategies.patch.patch_extractor import get_patch_transform
from app.strategies.resize.resize_strategy import predict_image
from app.strategies.multiscale.multiscale_extractor import (
    extract_object_context_patches,
    extract_native_texture_patches,
)


def _predict_patch_batch(
    model: torch.nn.Module,
    device: torch.device,
    patches: List[Image.Image],
    batch_size: int = INFERENCE_BATCH_SIZE
) -> Tuple[List[float], List[float]]:
    """
    Executes batched inference on a list of 32x32 PIL Image patches using frozen model.
    """
    if not patches:
        return [], []

    patch_transform = get_patch_transform()
    fake_probs: List[float] = []
    real_probs: List[float] = []

    for i in range(0, len(patches), batch_size):
        batch_patches = patches[i:i + batch_size]
        batch_tensors = torch.stack([patch_transform(p) for p in batch_patches]).to(device)

        with torch.inference_mode():
            logits = model(batch_tensors)
            probs = F.softmax(logits, dim=1)

        fake_probs.extend([float(p.item()) for p in probs[:, 0]])
        real_probs.extend([float(p.item()) for p in probs[:, 1]])

    return fake_probs, real_probs


def _calculate_branch_statistics(
    fake_probs: List[float],
    real_probs: List[float],
    coords: List[Dict[str, int]],
    source_name: str,
    top_k_ratio: float = MULTISCALE_TOP_K_RATIO
) -> Dict[str, Any]:
    """
    Calculates detailed patch statistics: mean, median, max, top-k mean, std, fake/real counts.
    """
    patch_count = len(fake_probs)
    if patch_count == 0:
        return {
            "fake_probability": 0.5,
            "real_probability": 0.5,
            "label": "UNCERTAIN",
            "confidence": 0.5,
            "patch_count": 0,
            "mean_fake_probability": 0.5,
            "median_fake_probability": 0.5,
            "max_fake_probability": 0.5,
            "top_k_mean_fake_probability": 0.5,
            "fake_patch_count": 0,
            "real_patch_count": 0,
            "fake_patch_ratio": 0.0,
            "patch_std": 0.0,
            "patch_predictions": []
        }

    arr_fake = np.array(fake_probs, dtype=np.float32)
    mean_fake = float(np.mean(arr_fake))
    median_fake = float(np.median(arr_fake))
    max_fake = float(np.max(arr_fake))
    patch_std = float(np.std(arr_fake))

    k_count = max(1, int(round(patch_count * top_k_ratio)))
    top_k_sorted = sorted(fake_probs, reverse=True)[:k_count]
    top_k_mean_fake = float(np.mean(top_k_sorted))

    fake_patch_count = sum(1 for p in fake_probs if p >= 0.50)
    real_patch_count = patch_count - fake_patch_count
    fake_patch_ratio = float(fake_patch_count / patch_count)

    # Individual patch records for debugging/visualization
    patch_records = []
    for fp, rp, coord in zip(fake_probs, real_probs, coords):
        lbl = "FAKE" if fp >= 0.50 else "REAL"
        patch_records.append({
            "x": coord["x"],
            "y": coord["y"],
            "width": coord["width"],
            "height": coord["height"],
            "fake_probability": round(fp, 4),
            "real_probability": round(rp, 4),
            "predicted_label": lbl,
            "confidence": round(fp if lbl == "FAKE" else rp, 4),
            "source": source_name
        })

    branch_fake_prob = mean_fake
    branch_real_prob = 1.0 - branch_fake_prob
    branch_label = "FAKE" if branch_fake_prob >= 0.50 else "REAL"
    branch_conf = branch_fake_prob if branch_label == "FAKE" else branch_real_prob

    return {
        "fake_probability": round(branch_fake_prob, 4),
        "real_probability": round(branch_real_prob, 4),
        "label": branch_label,
        "confidence": round(branch_conf, 4),
        "patch_count": patch_count,
        "mean_fake_probability": round(mean_fake, 4),
        "median_fake_probability": round(median_fake, 4),
        "max_fake_probability": round(max_fake, 4),
        "top_k_mean_fake_probability": round(top_k_mean_fake, 4),
        "fake_patch_count": fake_patch_count,
        "real_patch_count": real_patch_count,
        "fake_patch_ratio": round(fake_patch_ratio, 4),
        "patch_std": round(patch_std, 4),
        "patch_predictions": patch_records
    }


def predict_image_multiscale(
    image: Image.Image,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    global_weight: float = MULTISCALE_GLOBAL_WEIGHT,
    context_weight: float = MULTISCALE_CONTEXT_WEIGHT,
    texture_weight: float = MULTISCALE_TEXTURE_WEIGHT,
    fake_threshold: float = MULTISCALE_FAKE_THRESHOLD,
    real_threshold: float = MULTISCALE_REAL_THRESHOLD,
    max_native_patches: int = MAX_NATIVE_PATCHES
) -> Dict[str, Any]:
    """
    Executes Multi-Scale Inference & Evidence Generation across 3 spatial branches:
    1. Global Branch (32x32 resized whole image)
    2. Object/Context Branch (128x128 resized -> 16 patches)
    3. Native Texture Branch (32x32 native crops)

    Combines scores with normalized weighted fusion and returns structured evidence.
    """
    if model is None or device is None:
        loaded_model, loaded_device = load_model()
        model = model or loaded_model
        device = device or loaded_device

    # ------------------------------------------------------------------
    # 1. Global Branch (Original 32x32 baseline)
    # ------------------------------------------------------------------
    global_res = predict_image(image, model=model, device=device)
    global_fake_prob = global_res["fake_probability"]
    global_real_prob = global_res["real_probability"]

    # ------------------------------------------------------------------
    # 2. Object / Context Branch (128x128 -> 16 patches)
    # ------------------------------------------------------------------
    ctx_patches, ctx_coords_128, ctx_coords_orig = extract_object_context_patches(image)
    ctx_fake_probs, ctx_real_probs = _predict_patch_batch(model, device, ctx_patches)
    ctx_branch_stats = _calculate_branch_statistics(
        ctx_fake_probs, ctx_real_probs, ctx_coords_orig, source_name="object_context"
    )

    # ------------------------------------------------------------------
    # 3. Native Texture Branch (32x32 native crops)
    # ------------------------------------------------------------------
    tex_patches, tex_coords_orig, skipped_count = extract_native_texture_patches(
        image, max_patches=max_native_patches
    )
    tex_fake_probs, tex_real_probs = _predict_patch_batch(model, device, tex_patches)
    tex_branch_stats = _calculate_branch_statistics(
        tex_fake_probs, tex_real_probs, tex_coords_orig, source_name="native_texture"
    )

    # ------------------------------------------------------------------
    # 4. Transparent Weighted Score Fusion & Re-normalization
    # ------------------------------------------------------------------
    active_branches = []
    weights_raw = []
    fake_probs_active = []

    # Global branch is always available
    active_branches.append("global")
    weights_raw.append(global_weight)
    fake_probs_active.append(global_fake_prob)

    if ctx_branch_stats["patch_count"] > 0:
        active_branches.append("object_context")
        weights_raw.append(context_weight)
        fake_probs_active.append(ctx_branch_stats["fake_probability"])

    if tex_branch_stats["patch_count"] > 0:
        active_branches.append("native_texture")
        weights_raw.append(texture_weight)
        fake_probs_active.append(tex_branch_stats["fake_probability"])

    # Re-normalize weights over available active branches
    total_raw_weight = sum(weights_raw)
    norm_weights = [w / total_raw_weight for w in weights_raw]

    final_fake_prob = float(sum(p * w for p, w in zip(fake_probs_active, norm_weights)))
    final_real_prob = float(1.0 - final_fake_prob)

    # Branch Disagreement & Variance
    branch_disagreement = float(np.std(fake_probs_active)) if len(fake_probs_active) > 1 else 0.0

    # ------------------------------------------------------------------
    # 5. Tri-State Decision Logic (REAL, FAKE, UNCERTAIN)
    # ------------------------------------------------------------------
    if branch_disagreement >= 0.25 and (0.35 <= final_fake_prob <= 0.65):
        final_label = "UNCERTAIN"
    elif final_fake_prob >= fake_threshold:
        final_label = "FAKE"
    elif final_fake_prob <= real_threshold:
        final_label = "REAL"
    else:
        final_label = "UNCERTAIN"

    final_confidence = final_fake_prob if final_label == "FAKE" else (
        final_real_prob if final_label == "REAL" else float(max(final_fake_prob, final_real_prob))
    )

    # ------------------------------------------------------------------
    # 6. Highlighted Inference Regions (Suspicious Patches >= 0.50)
    # ------------------------------------------------------------------
    all_patch_records = ctx_branch_stats["patch_predictions"] + tex_branch_stats["patch_predictions"]
    highlighted_regions = sorted(
        [p for p in all_patch_records if p["fake_probability"] >= 0.50],
        key=lambda x: x["fake_probability"],
        reverse=True
    )[:10]  # Top 10 suspicious inference regions

    # Explanatory Text Generator
    if final_label == "UNCERTAIN":
        explanation = f"Multi-scale branches show disagreement (variance: {branch_disagreement:.2f}). Further review is recommended."
    elif final_label == "FAKE":
        explanation = f"Multi-scale analysis detected consistent synthetic AI artifacts (Final Fake Prob: {final_fake_prob*100:.1f}%)."
    else:
        explanation = f"Multi-scale analysis confirmed authentic photographic patterns (Final Real Prob: {final_real_prob*100:.1f}%)."

    return {
        "label": final_label,
        "confidence": round(final_confidence, 4),
        "fake_probability": round(final_fake_prob, 4),
        "real_probability": round(final_real_prob, 4),
        "inference_mode": "multiscale",
        "explanation": explanation,

        "analysis": {
            "global": {
                "label": global_res["label"],
                "confidence": round(global_res["confidence"], 4),
                "fake_probability": round(global_fake_prob, 4),
                "real_probability": round(global_real_prob, 4)
            },
            "object_context": ctx_branch_stats,
            "native_texture": tex_branch_stats,
            "fusion": {
                "active_branches": active_branches,
                "global_weight": round(norm_weights[0], 4),
                "object_context_weight": round(norm_weights[1], 4) if "object_context" in active_branches else 0.0,
                "native_texture_weight": round(norm_weights[-1], 4) if "native_texture" in active_branches else 0.0,
                "final_fake_probability": round(final_fake_prob, 4),
                "final_real_probability": round(final_real_prob, 4),
                "branch_disagreement": round(branch_disagreement, 4)
            },
            "highlighted_regions": highlighted_regions
        },

        "confidence_info": interpret_confidence(final_confidence)
    }
