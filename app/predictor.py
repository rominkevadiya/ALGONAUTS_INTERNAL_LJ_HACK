"""
SignalScope Prediction Engine
Handles image preprocessing, inference execution, and post-processing.
"""

from typing import Dict, Any, List
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd

try:
    from app.config import (
        IMAGE_SIZE,
        IMAGENET_MEAN,
        IMAGENET_STD,
        CLASS_MAPPING,
        HIGH_CONFIDENCE_THRESHOLD,
        MODERATE_CONFIDENCE_THRESHOLD
    )
    from app.model_loader import load_model, get_device
except (ImportError, ModuleNotFoundError):
    from config import (
        IMAGE_SIZE,
        IMAGENET_MEAN,
        IMAGENET_STD,
        CLASS_MAPPING,
        HIGH_CONFIDENCE_THRESHOLD,
        MODERATE_CONFIDENCE_THRESHOLD
    )
    from model_loader import load_model, get_device


def get_inference_transform() -> transforms.Compose:
    """
    Returns torchvision transforms matching training preprocessing pipeline:
    - Resize to (32, 32) using BICUBIC interpolation
    - ToTensor (scale [0, 255] -> [0.0, 1.0])
    - Normalize with ImageNet mean and std
    """
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Safely preprocesses a PIL Image:
    1. Validates and converts image to 3-channel RGB (with proper white background compositing for RGBA/transparency).
    2. Applies evaluation transforms with bicubic interpolation.
    3. Adds a batch dimension (shape: 1 x 3 x 32 x 32).
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a valid PIL Image instance.")

    try:
        # Auto-rotate image based on EXIF tag if present
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    # Ensure 3-channel RGB representation with proper alpha compositing
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        alpha_mask = image.convert("RGBA").split()[-1]
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image.convert("RGBA"), mask=alpha_mask)
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")

    transform = get_inference_transform()
    tensor_image = transform(image)
    # Add batch dimension
    return tensor_image.unsqueeze(0)


def interpret_confidence(confidence: float) -> Dict[str, str]:
    """
    Categorizes model confidence score into intuitive levels for UI display.
    """
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return {
            "level": "High Confidence",
            "status": "success" if confidence >= HIGH_CONFIDENCE_THRESHOLD else "warning",
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


def predict_image(
    image: Image.Image,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None
) -> Dict[str, Any]:
    """
    Performs AI vs REAL inference on a single PIL Image.

    Returns:
        Dict containing:
            - label: "FAKE" or "REAL"
            - confidence: float (0.0 to 1.0)
            - fake_probability: float (0.0 to 1.0)
            - real_probability: float (0.0 to 1.0)
            - confidence_info: Dict with interpretability metrics
    """
    if model is None or device is None:
        loaded_model, loaded_device = load_model()
        model = model or loaded_model
        device = device or loaded_device

    input_tensor = preprocess_image(image).to(device)

    with torch.no_grad():
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
        "confidence_info": interpret_confidence(confidence)
    }


def predict_batch(
    images_dict: Dict[str, Image.Image],
    model: torch.nn.Module | None = None,
    device: torch.device | None = None
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
            res = predict_image(img, model=model, device=device)
            results.append({
                "Filename": filename,
                "Prediction": res["label"],
                "Confidence": f"{res['confidence'] * 100:.2f}%",
                "Fake Probability": f"{res['fake_probability'] * 100:.2f}%",
                "Real Probability": f"{res['real_probability'] * 100:.2f}%",
                "Raw Confidence": res["confidence"]
            })
        except Exception as e:
            results.append({
                "Filename": filename,
                "Prediction": "ERROR",
                "Confidence": "N/A",
                "Fake Probability": "N/A",
                "Real Probability": "N/A",
                "Raw Confidence": 0.0
            })

    return pd.DataFrame(results)
