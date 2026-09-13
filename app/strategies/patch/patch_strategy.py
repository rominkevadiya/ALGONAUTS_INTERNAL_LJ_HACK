from typing import Dict, Any
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
from app.model_loader import load_model
from app.diagnostics.entropy import interpret_confidence
from app.strategies.patch.patch_extractor import (
    prepare_image,
    get_patch_transform,
    extract_native_patches,
)


def predict_image_patch_vote(
    image: Image.Image,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    n_patches: int = PATCH_N,
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

    if model is None or device is None:
        loaded_model, loaded_device = load_model()
        model = model or loaded_model
        device = device or loaded_device

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
    
    # Calculate patch luminances (brightness) to distinguish sensor noise from AI artifacts
    patch_luminances = [float(np.mean(np.array(p, dtype=np.float32))) for p in patches]
    lit_fake_probs = [fp for fp, lum in zip(fake_probs_list, patch_luminances) if lum >= 45.0]
    lit_patch_fake = float(np.mean(lit_fake_probs)) if lit_fake_probs else float(np.mean(fake_probs_list))

    # Top-K (top 25% highest fake probability patches)
    k_count = max(1, len(fake_probs_list) // 4)
    top_k_fake_probs = sorted(fake_probs_list, reverse=True)[:k_count]
    top_k_patch_fake = float(np.mean(top_k_fake_probs))

    if aggregation == "mean":
        fake_prob = float(fake_probs_t.mean().item())
        real_prob = float(real_probs_t.mean().item())
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

    label = "FAKE" if fake_prob > real_prob else "REAL"
    confidence = fake_prob if label == "FAKE" else real_prob

    return {
        "label": label,
        "confidence": confidence,
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "patch_count": len(patches),
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
