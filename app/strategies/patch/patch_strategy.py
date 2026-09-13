from typing import Dict, Any
import math
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np

from app.config import (
    IMAGE_SIZE,
    PATCH_N,
    PATCH_AGGREGATION_DEFAULT,
    PATCH_AGGREGATION_METHODS,
)
from app.model_loader import load_model, resolve_model_device
from app.diagnostics.entropy import interpret_confidence
from app.strategies.patch.patch_extractor import (
    prepare_image,
    get_patch_transform,
    extract_native_patches,
)
from app.strategies.base_strategy import BaseStrategy, validate_strategy_output
from app.strategies.strategy_registry import register_strategy


def predict_image_patch_vote(
    image: Image.Image,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    n_patches: int = 0,  # 0 enables dynamic patch count based on resolution
    seed: int = 42,
    aggregation: str = PATCH_AGGREGATION_DEFAULT
) -> Dict[str, Any]:
    """
    Native-resolution patch-based inference.
    Extracts N native 32x32 crops, batches them into a single tensor, and aggregates predictions.
    Supports aggregation: 'mean', 'median', 'majority', 'logit_mean', 'max', 'top_k'.
    """
    if aggregation not in PATCH_AGGREGATION_METHODS:
        raise ValueError(f"Invalid aggregation method '{aggregation}'. Supported: {PATCH_AGGREGATION_METHODS}")

    model, device = resolve_model_device(model, device)

    clean_img = prepare_image(image)
    w, h = clean_img.size

    # Fall back to resize mode if image is smaller than 32x32 patch size
    if w < IMAGE_SIZE[0] or h < IMAGE_SIZE[1]:
        from app.strategies.resize.resize_strategy import predict_image
        res = predict_image(clean_img, model=model, device=device)
        res["patch_count"] = 1
        res["aggregation"] = aggregation
        res["patch_fake_probs"] = [res["fake_probability"]]
        res["patch_real_probs"] = [res["real_probability"]]
        res["patch_coordinates"] = [(0, 0, w, h)]
        res["max_patch_fake_prob"] = res["fake_probability"]
        res["min_patch_fake_prob"] = res["fake_probability"]
        res["top_k_patch_fake_prob"] = res["fake_probability"]
        res["lit_patch_fake_prob"] = res["fake_probability"]
        res["inference_mode"] = "resize_fallback"
        return res

    patches, coords = extract_native_patches(clean_img, patch_size=IMAGE_SIZE, n_patches=n_patches, seed=seed)
    actual_n_patches = len(patches)
    patch_transform = get_patch_transform()
    patch_tensors = torch.stack([patch_transform(p) for p in patches]).to(device)

    with torch.inference_mode():
        logits = model(patch_tensors)
        probabilities = F.softmax(logits, dim=1)

    fake_probs_t = probabilities[:, 0]
    real_probs_t = probabilities[:, 1]

    fake_probs_list = [float(p.item()) for p in fake_probs_t]
    real_probs_list = [float(p.item()) for p in real_probs_t]

    max_patch_fake = float(np.max(fake_probs_list))
    min_patch_fake = float(np.min(fake_probs_list))
    
    # Adaptive luminance threshold to distinguish sensor noise from AI artifacts
    # Threshold is 40% of the image's overall mean brightness, clamped [30, 80]
    img_array = np.array(clean_img.convert("L"), dtype=np.float32)
    img_mean_brightness = float(np.mean(img_array))
    lit_threshold = max(30.0, min(80.0, img_mean_brightness * 0.40))

    patch_luminances = [float(np.mean(np.array(p, dtype=np.float32))) for p in patches]
    lit_fake_probs = [fp for fp, lum in zip(fake_probs_list, patch_luminances) if lum >= lit_threshold]
    lit_patch_fake = float(np.mean(lit_fake_probs)) if lit_fake_probs else float(np.mean(fake_probs_list))

    # Top-K (top 25% highest fake probability patches)
    k_count = max(1, len(fake_probs_list) // 4)
    top_k_fake_probs = sorted(fake_probs_list, reverse=True)[:k_count]
    top_k_patch_fake = float(np.mean(top_k_fake_probs))

    # Calculate center-weighted aggregation (salient zone weighting)
    # Patches closer to the center get up to 2x weight
    cx, cy = w / 2, h / 2
    max_dist = math.sqrt(cx**2 + cy**2)
    weights = []
    for (px1, py1, px2, py2) in coords:
        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
        dist = math.sqrt((pcx - cx)**2 + (pcy - cy)**2)
        # Weight goes from 2.0 (at center) to 1.0 (at corners)
        weight = 1.0 + (1.0 - (dist / max_dist))
        weights.append(weight)
    
    total_weight = sum(weights)
    weighted_mean_fake = sum(fp * wt for fp, wt in zip(fake_probs_list, weights)) / total_weight
    weighted_mean_real = sum(rp * wt for rp, wt in zip(real_probs_list, weights)) / total_weight

    if aggregation == "mean":
        fake_prob = weighted_mean_fake
        real_prob = weighted_mean_real
    elif aggregation == "median":
        fake_prob = float(fake_probs_t.median().item())
        real_prob = float(real_probs_t.median().item())
        total = fake_prob + real_prob
        fake_prob /= total
        real_prob /= total
    elif aggregation == "majority":
        fake_votes = sum(1 for p in fake_probs_list if p > 0.5)
        total_votes = len(fake_probs_list)
        fake_prob = fake_votes / total_votes
        real_prob = 1.0 - fake_prob
    elif aggregation == "logit_mean":
        avg_logits = logits.mean(dim=0, keepdim=True)
        avg_probs = F.softmax(avg_logits, dim=1).squeeze(0)
        fake_prob = float(avg_probs[0].item())
        real_prob = float(avg_probs[1].item())
    elif aggregation == "max":
        fake_prob = max_patch_fake
        real_prob = 1.0 - fake_prob
    elif aggregation == "top_k":
        fake_prob = top_k_patch_fake
        real_prob = 1.0 - fake_prob
    else:
        fake_prob = weighted_mean_fake
        real_prob = weighted_mean_real

    label = "FAKE" if fake_prob > real_prob else "REAL"
    confidence = fake_prob if label == "FAKE" else real_prob

    res = {
        "label": label,
        "confidence": confidence,
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "patch_count": actual_n_patches,
        "aggregation": aggregation,
        "patch_fake_probs": fake_probs_list,
        "patch_real_probs": real_probs_list,
        "max_patch_fake_prob": max_patch_fake,
        "min_patch_fake_prob": min_patch_fake,
        "top_k_patch_fake_prob": top_k_patch_fake,
        "lit_patch_fake_prob": lit_patch_fake,
        "patch_coordinates": coords,
        "inference_mode": "patch",
        "confidence_info": interpret_confidence(confidence)
    }
    
    # Store internal patch array strictly for Hybrid strategy to re-evaluate if needed
    res["_patches"] = patches
    return validate_strategy_output(res, strategy_name="patch")





class PatchStrategy(BaseStrategy):
    """Concrete BaseStrategy implementation for Native Patch Voting Strategy."""

    @property
    def name(self) -> str:
        return "patch"

    @property
    def display_name(self) -> str:
        return "Patch Grid Analysis"

    @property
    def description(self) -> str:
        return "Extracts native-resolution 32x32 patches across the image grid and aggregates predictions."

    def predict(
        self,
        image: Image.Image,
        model: torch.nn.Module | None = None,
        device: torch.device | None = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        return predict_image_patch_vote(image, model=model, device=device, **kwargs)


# Register strategy with StrategyRegistry
register_strategy(PatchStrategy())

