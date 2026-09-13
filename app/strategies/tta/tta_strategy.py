import math
from typing import List, Dict, Any, Optional
from PIL import Image, ImageEnhance
import torch
import torch.nn.functional as F
import numpy as np

from app.model_loader import load_model, resolve_model_device
from app.diagnostics.entropy import interpret_confidence
from app.strategies.patch.patch_extractor import prepare_image, get_inference_transform
from app.strategies.base_strategy import BaseStrategy, validate_strategy_output
from app.strategies.strategy_registry import register_strategy


def _apply_brightness(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)


def _apply_contrast(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(factor)


def _rotate(img: Image.Image, degrees: float) -> Image.Image:
    return img.rotate(degrees, expand=False)


def predict_image_tta(
    image: Image.Image,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    """
    Enhanced Test-Time Augmentation (TTA) inference.

    Generates 8 semantically meaningful geometric and photometric views:
      1. Original
      2. Horizontal Flip (left-right mirror)
      3. Center crop (10% margin removed)
      4. Brightness +15% (simulate slightly overexposed shot)
      5. Brightness -15% (simulate slightly underexposed shot)
      6. Contrast +20% (sharpen edges — exposes AI over-smoothing)
      7. 90° clockwise rotation
      8. 45° clockwise rotation (tests equivariance)

    Aggregates using inverse-entropy weighting: views where the model is
    most decisive (lowest entropy) contribute more to the final prediction.
    Also reports a self-consistency flag when original vs. H-flip disagree.
    """
    model, device = resolve_model_device(model, device)

    clean_img = prepare_image(image)
    w, h = clean_img.size

    # Build 8 augmented views with descriptive names
    views: List[tuple[str, Image.Image]] = [
        ("original",          clean_img),
        ("hflip",             clean_img.transpose(Image.FLIP_LEFT_RIGHT)),
        ("center_crop",       clean_img.crop((int(w * 0.10), int(h * 0.10), int(w * 0.90), int(h * 0.90)))),
        ("bright+15",         _apply_brightness(clean_img, 1.15)),
        ("bright-15",         _apply_brightness(clean_img, 0.85)),
        ("contrast+20",       _apply_contrast(clean_img, 1.20)),
        ("rotate90",          _rotate(clean_img, 90)),
        ("rotate45",          _rotate(clean_img, 45)),
    ]

    view_names = [name for name, _ in views]
    view_imgs = [img for _, img in views]

    transform = get_inference_transform()
    tensors = torch.stack([transform(v) for v in view_imgs]).to(device)

    with torch.inference_mode():
        logits = model(tensors)
        probs = F.softmax(logits, dim=1)

    fake_probs: List[float] = [float(p.item()) for p in probs[:, 0]]
    real_probs: List[float] = [float(p.item()) for p in probs[:, 1]]

    # Inverse-entropy weighting: views with lower entropy contribute more
    # H(p) = -p*log2(p) - (1-p)*log2(1-p), max=1 at p=0.5
    def _entropy(fp: float) -> float:
        p = max(1e-9, min(1 - 1e-9, fp))
        q = 1.0 - p
        return -(p * math.log2(p) + q * math.log2(q))

    entropies = [_entropy(fp) for fp in fake_probs]
    # Weight = 1 - H_norm (so low entropy = high weight)
    weights = [max(0.01, 1.0 - e) for e in entropies]
    total_weight = sum(weights)
    norm_weights = [w / total_weight for w in weights]

    weighted_fake = float(sum(fp * wt for fp, wt in zip(fake_probs, norm_weights)))
    weighted_real = 1.0 - weighted_fake

    # Arithmetic mean for comparison
    mean_fake = float(np.mean(fake_probs))
    mean_real = float(np.mean(real_probs))
    std_fake = float(np.std(fake_probs))

    # Self-consistency check: original vs. H-flip prediction agreement
    orig_label = "FAKE" if fake_probs[0] > 0.5 else "REAL"
    hflip_label = "FAKE" if fake_probs[1] > 0.5 else "REAL"
    self_consistent = orig_label == hflip_label

    label = "FAKE" if weighted_fake > weighted_real else "REAL"
    confidence = weighted_fake if label == "FAKE" else weighted_real

    res = {
        "label": label,
        "confidence": confidence,
        "fake_probability": weighted_fake,
        "real_probability": weighted_real,
        "mean_fake_probability": mean_fake,
        "mean_real_probability": mean_real,
        "tta_view_count": len(views),
        "tta_view_names": view_names,
        "tta_fake_probs": fake_probs,
        "tta_real_probs": real_probs,
        "tta_weights": norm_weights,
        "tta_entropies": entropies,
        "tta_std": std_fake,
        "self_consistent": self_consistent,
        "orig_label": orig_label,
        "hflip_label": hflip_label,
        "inference_mode": "tta",
        "confidence_info": interpret_confidence(confidence)
    }
    return validate_strategy_output(res, strategy_name="tta")




class TTAStrategy(BaseStrategy):
    """Concrete BaseStrategy implementation for Test-Time Augmentation Strategy."""

    @property
    def name(self) -> str:
        return "tta"

    @property
    def display_name(self) -> str:
        return "Test-Time Augmentation (8-View TTA)"

    @property
    def description(self) -> str:
        return "Evaluates 8 geometric/photometric views with inverse-entropy weighting for robust inference."

    def predict(
        self,
        image: Image.Image,
        model: torch.nn.Module | None = None,
        device: torch.device | None = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        return predict_image_tta(image, model=model, device=device)


# Register strategy with StrategyRegistry
register_strategy(TTAStrategy())

