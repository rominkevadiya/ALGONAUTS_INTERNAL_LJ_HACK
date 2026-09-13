from typing import Dict, Any
from PIL import Image
import torch
import torch.nn.functional as F

from app.config import CLASS_MAPPING
from app.model_loader import load_model
from app.diagnostics.entropy import interpret_confidence
from app.strategies.patch.patch_extractor import preprocess_image


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

    res = {
        "label": predicted_label,
        "confidence": confidence,
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "inference_mode": "resize",
        "confidence_info": interpret_confidence(confidence)
    }
    return validate_strategy_output(res, strategy_name="resize")


from app.strategies.base_strategy import BaseStrategy, validate_strategy_output
from app.strategies.strategy_registry import register_strategy


class ResizeStrategy(BaseStrategy):
    """Concrete BaseStrategy implementation for Baseline Single-Pass Resize Strategy."""

    @property
    def name(self) -> str:
        return "resize"

    @property
    def display_name(self) -> str:
        return "Single-Pass (32x32 Resize Baseline)"

    @property
    def description(self) -> str:
        return "Resizes the entire image directly to 32x32 and executes single-pass inference."

    def predict(
        self,
        image: Image.Image,
        model: torch.nn.Module | None = None,
        device: torch.device | None = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        return predict_image(image, model=model, device=device)


# Register strategy with StrategyRegistry
register_strategy(ResizeStrategy())

