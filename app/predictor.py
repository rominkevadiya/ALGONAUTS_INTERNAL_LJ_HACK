"""
SignalScope Prediction Engine
Handles image preprocessing, inference execution, post-processing, and multi-strategy diagnostics.
"""

from typing import Dict, Any, List, Tuple
import math
import random
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms

try:
    from app.config import (
        IMAGE_SIZE,
        IMAGENET_MEAN,
        IMAGENET_STD,
        CLASS_MAPPING,
        HIGH_CONFIDENCE_THRESHOLD,
        MODERATE_CONFIDENCE_THRESHOLD,
        DEFAULT_INFERENCE_MODE,
        PATCH_N,
        PATCH_THRESHOLD_PX,
        PATCH_AGGREGATION_DEFAULT,
        PATCH_AGGREGATION_METHODS,
        INFERENCE_MODES,
        ENTROPY_LOW_THRESHOLD,
        ENTROPY_HIGH_THRESHOLD,
        HYBRID_STRONG_DIFF,
        HYBRID_PARTIAL_DIFF
    )
    from app.model_loader import load_model, get_device
except (ImportError, ModuleNotFoundError):
    from config import (
        IMAGE_SIZE,
        IMAGENET_MEAN,
        IMAGENET_STD,
        CLASS_MAPPING,
        HIGH_CONFIDENCE_THRESHOLD,
        MODERATE_CONFIDENCE_THRESHOLD,
        DEFAULT_INFERENCE_MODE,
        PATCH_N,
        PATCH_THRESHOLD_PX,
        PATCH_AGGREGATION_DEFAULT,
        PATCH_AGGREGATION_METHODS,
        INFERENCE_MODES,
        ENTROPY_LOW_THRESHOLD,
        ENTROPY_HIGH_THRESHOLD,
        HYBRID_STRONG_DIFF,
        HYBRID_PARTIAL_DIFF
    )
    from model_loader import load_model, get_device


def prepare_image(image: Image.Image) -> Image.Image:
    """
    Single unified helper for image validation, EXIF correction, transparency compositing, and RGB conversion.
    Does NOT mutate the original image object.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a valid PIL Image instance.")

    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    # Handle RGBA, LA, or Palette images with transparency using white background compositing
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        alpha_mask = image.convert("RGBA").split()[-1]
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image.convert("RGBA"), mask=alpha_mask)
        return background
    elif image.mode != "RGB":
        return image.convert("RGB")
    
    return image.copy()


def get_inference_transform() -> transforms.Compose:
    """
    Returns torchvision transforms matching training preprocessing pipeline:
    - Resize to IMAGE_SIZE (32, 32) using BICUBIC interpolation
    - ToTensor (scale [0, 255] -> [0.0, 1.0])
    - Normalize with ImageNet mean and std
    """
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_patch_transform() -> transforms.Compose:
    """
    Returns torchvision transforms for native resolution patches (no resize):
    - ToTensor
    - Normalize with ImageNet mean and std
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Safely preprocesses a PIL Image for standard 32x32 resize inference:
    1. Validates and converts image to 3-channel RGB via prepare_image.
    2. Applies evaluation transforms with bicubic interpolation.
    3. Adds a batch dimension (shape: 1 x 3 x 32 x 32).
    """
    clean_image = prepare_image(image)
    transform = get_inference_transform()
    tensor_image = transform(clean_image)
    return tensor_image.unsqueeze(0)


def extract_native_patches(
    image: Image.Image,
    patch_size: Tuple[int, int] = IMAGE_SIZE,
    n_patches: int = PATCH_N,
    seed: int = 42
) -> Tuple[List[Image.Image], List[Tuple[int, int, int, int]]]:
    """
    Extracts native-resolution patch crops from the original image.
    Uses a local deterministic random generator (rng = random.Random(seed)).
    Combines center crop, 4 corner crops, grid crops, and random crops.
    """
    clean_image = prepare_image(image)
    w, h = clean_image.size
    pw, ph = patch_size

    if w < pw or h < ph:
        return [clean_image], [(0, 0, w, h)]

    rng = random.Random(seed)
    crops: List[Image.Image] = []
    coords: List[Tuple[int, int, int, int]] = []
    seen_coords = set()

    def add_crop(x1: int, y1: int):
        x1 = max(0, min(x1, w - pw))
        y1 = max(0, min(y1, h - ph))
        box = (x1, y1, x1 + pw, y1 + ph)
        if box not in seen_coords:
            seen_coords.add(box)
            crops.append(clean_image.crop(box))
            coords.append(box)

    # 1. Center crop
    add_crop((w - pw) // 2, (h - ph) // 2)

    # 2. Four corners
    add_crop(0, 0)
    add_crop(w - pw, 0)
    add_crop(0, h - ph)
    add_crop(w - pw, h - ph)

    # 3. Grid crops
    grid_cols = max(2, int(math.sqrt(n_patches)))
    grid_rows = max(2, int(math.sqrt(n_patches)))
    x_steps = np.linspace(0, max(0, w - pw), grid_cols, dtype=int)
    y_steps = np.linspace(0, max(0, h - ph), grid_rows, dtype=int)

    for gx in x_steps:
        for gy in y_steps:
            add_crop(int(gx), int(gy))

    # 4. Fill remaining with random crops
    attempts = 0
    max_attempts = n_patches * 10
    while len(crops) < n_patches and attempts < max_attempts:
        rx = rng.randint(0, w - pw)
        ry = rng.randint(0, h - ph)
        add_crop(rx, ry)
        attempts += 1

    # If still fewer (e.g. image barely larger than 32x32), duplicate with small offsets if needed
    while len(crops) < n_patches:
        rx = rng.randint(0, max(0, w - pw))
        ry = rng.randint(0, max(0, h - ph))
        box = (rx, ry, rx + pw, ry + ph)
        crops.append(clean_image.crop(box))
        coords.append(box)

    return crops[:n_patches], coords[:n_patches]


def interpret_confidence(confidence: float) -> Dict[str, str]:
    """
    Categorizes model confidence score into intuitive levels for UI display.
    """
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return {
            "level": "High Confidence",
            "status": "success",
            "message": "Model is highly confident in this prediction."
        }
    elif confidence >= MODERATE_CONFIDENCE_THRESHOLD:
        return {
            "level": "Moderate Confidence",
            "status": "info",
            "message": "Model prediction has moderate certainty."
        }
    else:
        return {
            "level": "Low Confidence / Review Recommended",
            "status": "warning",
            "message": "Prediction score is near the decision boundary. Verification recommended."
        }


def compute_prediction_entropy(fake_prob: float, real_prob: float) -> Dict[str, Any]:
    """
    Computes normalized binary Shannon entropy:
    H(p) = -p log2(p) - (1-p) log2(1-p)
    Normalized H_norm = H(p) / log2(2) = H(p)
    """
    p_fake = max(1e-9, min(1.0 - 1e-9, float(fake_prob)))
    p_real = max(1e-9, min(1.0 - 1e-9, float(real_prob)))

    h = -(p_fake * math.log2(p_fake) + p_real * math.log2(p_real))
    norm_h = float(min(1.0, max(0.0, h)))

    if norm_h < ENTROPY_LOW_THRESHOLD:
        level = "Low Output Entropy"
        note = "Model prediction is highly decisive. Output probability distribution is concentrated."
    elif norm_h < ENTROPY_HIGH_THRESHOLD:
        level = "Moderate Output Entropy"
        note = "Model prediction shows moderate uncertainty."
    else:
        level = "High Output Entropy"
        note = "Model prediction is near the 50/50 decision boundary. High output uncertainty."

    return {
        "entropy": norm_h * math.log(2),  # Natural log entropy
        "normalized_entropy": norm_h,
        "uncertainty_level": level,
        "uncertainty_note": note + " (Note: Entropy measures model output certainty, not guaranteed correctness)."
    }


def compute_prediction_disagreement(probabilities: List[float]) -> Dict[str, Any]:
    """
    Calculates statistical disagreement metrics over a set of patch or TTA probabilities.
    """
    if not probabilities:
        return {
            "mean_fake_probability": 0.0,
            "median_fake_probability": 0.0,
            "std_fake_probability": 0.0,
            "min_fake_probability": 0.0,
            "max_fake_probability": 0.0,
            "range_fake_probability": 0.0,
            "fake_patch_count": 0,
            "real_patch_count": 0,
            "patch_agreement_pct": 100.0
        }

    probs = [float(p) for p in probabilities]
    n = len(probs)
    mean_p = float(np.mean(probs))
    med_p = float(np.median(probs))
    std_p = float(np.std(probs))
    min_p = float(np.min(probs))
    max_p = float(np.max(probs))
    range_p = max_p - min_p

    fake_count = sum(1 for p in probs if p > 0.5)
    real_count = n - fake_count
    majority_count = max(fake_count, real_count)
    agreement_pct = (majority_count / n) * 100.0

    return {
        "mean_fake_probability": mean_p,
        "median_fake_probability": med_p,
        "std_fake_probability": std_p,
        "min_fake_probability": min_p,
        "max_fake_probability": max_p,
        "range_fake_probability": range_p,
        "fake_patch_count": fake_count,
        "real_patch_count": real_count,
        "patch_agreement_pct": float(agreement_pct)
    }


def compute_fft_spectral_diagnostic(image: Image.Image) -> Dict[str, Any]:
    """
    Experimental 2D Fast Fourier Transform (FFT) spectral diagnostic.
    Converts image to floating-point grayscale, removes mean intensity, calculates 2D FFT,
    and analyzes radial frequency energy distribution.
    """
    clean_image = prepare_image(image)
    gray = np.array(clean_image.convert("L"), dtype=np.float32)
    
    # Remove mean intensity to eliminate DC spike
    gray_zero_mean = gray - np.mean(gray)
    
    fft = np.fft.fft2(gray_zero_mean)
    shifted = np.fft.fftshift(fft)
    magnitude = np.abs(shifted)
    log_power = np.log1p(magnitude ** 2)

    h, w = log_power.shape
    cy, cx = h // 2, w // 2
    max_radius = min(cy, cx)

    if max_radius < 4:
        return {
            "spectral_score": 0.0,
            "low_frequency_energy": 0.0,
            "high_frequency_energy": 0.0,
            "spectral_slope": 0.0,
            "diagnostic_label": "Low spectral irregularity",
            "interpretation": "Experimental frequency-domain diagnostic only. Affected by image content and resolution."
        }

    y_grid, x_grid = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)

    low_freq_mask = dist_from_center <= (max_radius * 0.25)
    high_freq_mask = (dist_from_center > (max_radius * 0.25)) & (dist_from_center <= max_radius)

    low_energy = float(np.mean(log_power[low_freq_mask])) if np.any(low_freq_mask) else 1e-6
    high_energy = float(np.mean(log_power[high_freq_mask])) if np.any(high_freq_mask) else 1e-6

    ratio = high_energy / (low_energy + 1e-9)
    spectral_score = float(min(1.0, max(0.0, ratio * 3.5)))

    # Radial profile slope estimation
    radii = np.arange(1, max_radius)
    intensity_profile = [float(np.mean(log_power[(dist_from_center >= r-0.5) & (dist_from_center < r+0.5)])) for r in radii]
    valid_mask = ~np.isnan(intensity_profile)
    if np.sum(valid_mask) > 3:
        log_r = np.log(radii[valid_mask])
        log_i = np.array(intensity_profile)[valid_mask]
        slope, _ = np.polyfit(log_r, log_i, 1)
    else:
        slope = 0.0

    if spectral_score < 0.35:
        label = "Low spectral irregularity"
    elif spectral_score < 0.65:
        label = "Moderate spectral irregularity"
    else:
        label = "High spectral irregularity"

    return {
        "spectral_score": spectral_score,
        "low_frequency_energy": low_energy,
        "high_frequency_energy": high_energy,
        "high_to_low_ratio": float(ratio),
        "spectral_slope": float(slope),
        "diagnostic_label": label,
        "interpretation": "Experimental frequency-domain diagnostic only. Not a trained classifier probability."
    }


def predict_image(
    image: Image.Image,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None
) -> Dict[str, Any]:
    """
    Baseline PyTorch inference: Resizes entire image to (32, 32) and executes ResNet-50.
    Preserves exact original behavior and return signature.
    """
    if model is None or device is None:
        loaded_model, loaded_device = load_model()
        model = model or loaded_model
        device = device or loaded_device

    input_tensor = preprocess_image(image).to(device)

    with torch.inference_mode():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1).squeeze(0)

    fake_prob = float(probabilities[0].item())
    real_prob = float(probabilities[1].item())

    predicted_class_idx = int(torch.argmax(probabilities).item())
    predicted_label = CLASS_MAPPING.get(predicted_class_idx, "UNKNOWN")
    confidence = fake_prob if predicted_label == "FAKE" else real_prob

    return {
        "label": predicted_label,
        "confidence": confidence,
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "inference_mode": "resize",
        "confidence_info": interpret_confidence(confidence)
    }


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
        res = predict_image(clean_img, model=model, device=device)
        res["patch_count"] = 1
        res["aggregation"] = aggregation
        res["patch_fake_probs"] = [res["fake_probability"]]
        res["patch_real_probs"] = [res["real_probability"]]
        res["patch_coordinates"] = [(0, 0, w, h)]
        res["max_patch_fake_prob"] = res["fake_probability"]
        res["min_patch_fake_prob"] = res["fake_probability"]
        res["top_k_patch_fake_prob"] = res["fake_probability"]
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
        "patch_coordinates": coords,
        "inference_mode": "patch",
        "confidence_info": interpret_confidence(confidence)
    }


def predict_image_tta(
    image: Image.Image,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    include_patch_mode: bool = False
) -> Dict[str, Any]:
    """
    Test-Time Augmentation (TTA) inference across multiple geometric views without retraining.
    """
    if model is None or device is None:
        loaded_model, loaded_device = load_model()
        model = model or loaded_model
        device = device or loaded_device

    clean_img = prepare_image(image)
    w, h = clean_img.size

    views: List[Image.Image] = [
        clean_img,
        clean_img.transpose(Image.FLIP_LEFT_RIGHT),
        clean_img.transpose(Image.FLIP_TOP_BOTTOM),
        clean_img.crop((int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)))
    ]

    transform = get_inference_transform()
    tensors = torch.stack([transform(v) for v in views]).to(device)

    with torch.inference_mode():
        logits = model(tensors)
        probs = F.softmax(logits, dim=1)

    fake_probs = [float(p.item()) for p in probs[:, 0]]
    real_probs = [float(p.item()) for p in probs[:, 1]]

    mean_fake = float(np.mean(fake_probs))
    mean_real = float(np.mean(real_probs))
    std_fake = float(np.std(fake_probs))

    label = "FAKE" if mean_fake > mean_real else "REAL"
    confidence = mean_fake if label == "FAKE" else mean_real

    return {
        "label": label,
        "confidence": confidence,
        "fake_probability": mean_fake,
        "real_probability": mean_real,
        "tta_view_count": len(views),
        "tta_fake_probs": fake_probs,
        "tta_real_probs": real_probs,
        "tta_std": std_fake,
        "inference_mode": "tta",
        "confidence_info": interpret_confidence(confidence)
    }


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
    patch_fake = patch_res["fake_probability"]
    max_patch_fake = patch_res.get("max_patch_fake_prob", patch_fake)
    top_k_patch_fake = patch_res.get("top_k_patch_fake_prob", patch_fake)

    diff = abs(resize_fake - patch_fake)

    if resize_res["label"] == patch_res["label"]:
        if diff < HYBRID_STRONG_DIFF:
            agreement = "Strong Agreement"
        elif diff < HYBRID_PARTIAL_DIFF:
            agreement = "Partial Agreement"
        else:
            agreement = "Moderate Agreement"
    else:
        agreement = "Disagreement"

    # Balanced hybrid decision rules
    if agreement == "Disagreement":
        # Strategy Disagreement (e.g. Resize says REAL, Patch says FAKE due to dark webcam noise):
        # Balance probabilities between strategies so result reflects genuine uncertainty
        hybrid_fake = (resize_fake + patch_fake) / 2.0
        hybrid_real = 1.0 - hybrid_fake
        hybrid_label = "FAKE" if hybrid_fake > hybrid_real else "REAL"
        hybrid_confidence = hybrid_fake if hybrid_label == "FAKE" else hybrid_real
    elif resize_res["label"] == "REAL" and (top_k_patch_fake >= 0.65 or max_patch_fake >= 0.80 or (patch_fake - resize_fake) >= 0.30):
        # Localized AI artifacts detected by patch voting in a high-resolution image!
        # Factor in top-k patch probability
        hybrid_fake = float(max(patch_fake, (resize_fake + top_k_patch_fake) / 2.0))
        # If top_k_patch_fake is strong (> 0.70), elevate fake activation
        if top_k_patch_fake >= 0.70:
            hybrid_fake = float((hybrid_fake + top_k_patch_fake) / 2.0)
        hybrid_real = 1.0 - hybrid_fake
        hybrid_label = "FAKE" if hybrid_fake > hybrid_real else "REAL"
        hybrid_confidence = hybrid_fake if hybrid_label == "FAKE" else hybrid_real
    else:
        hybrid_label = patch_res["label"]
        hybrid_confidence = patch_res["confidence"]
        hybrid_fake = patch_fake
        hybrid_real = 1.0 - hybrid_fake

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
        "inference_mode": "hybrid",
        "confidence_info": interpret_confidence(hybrid_confidence)
    }


def predict_image_auto(
    image: Image.Image,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    mode: str = DEFAULT_INFERENCE_MODE,
    n_patches: int = PATCH_N,
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
        if min_dim >= PATCH_THRESHOLD_PX:
            selected_mode = "patch"
        else:
            selected_mode = "resize"

    if selected_mode == "resize":
        result = predict_image(clean_img, model=model, device=device)
    elif selected_mode == "patch":
        result = predict_image_patch_vote(
            clean_img, model=model, device=device, n_patches=n_patches, seed=seed, aggregation=aggregation
        )
    elif selected_mode == "hybrid":
        result = predict_image_hybrid(
            clean_img, model=model, device=device, n_patches=n_patches, seed=seed, aggregation=aggregation
        )
    elif selected_mode == "tta":
        result = predict_image_tta(clean_img, model=model, device=device)

    # Attach entropy, disagreement, and FFT diagnostics to the output dictionary
    entropy_info = compute_prediction_entropy(result["fake_probability"], result["real_probability"])
    result.update(entropy_info)

    if "patch_fake_probs" in result:
        disagreement_info = compute_prediction_disagreement(result["patch_fake_probs"])
        result["stability"] = disagreement_info

    result["fft_diagnostic"] = compute_fft_spectral_diagnostic(clean_img)
    result["image_dimensions"] = f"{w} x {h}"
    return result


def predict_batch(
    images_dict: Dict[str, Image.Image],
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    mode: str = DEFAULT_INFERENCE_MODE,
    n_patches: int = PATCH_N,
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

    for filename, img in images_dict.items():
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
        except Exception:
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

