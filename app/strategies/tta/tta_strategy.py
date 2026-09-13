from typing import List, Dict, Any
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np

from app.model_loader import load_model
from app.diagnostics.entropy import interpret_confidence
from app.strategies.patch.patch_extractor import prepare_image, get_inference_transform


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
