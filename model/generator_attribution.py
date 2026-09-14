import json
from typing import Any, Dict
import warnings

from PIL import Image

from app.api.gemini_gateway import generate_multimodal
from app.api.prompt_registry import generator_attribution_prompt

warnings.filterwarnings("ignore", message=".*Direct use of automatic function calling.*")

_FAMILIES = {"Diffusion", "GAN", "Unknown"}
_MODELS = {"Midjourney", "Stable Diffusion", "DALL-E", "StyleGAN", "Unknown"}


def predict_generator_attribution(image: Image.Image) -> Dict[str, Any]:
    """Predict the likely generator family using the centralized Gemini gateway."""
    version, prompt = generator_attribution_prompt()
    response = generate_multimodal(
        task_id=version,
        image=image,
        prompt=prompt,
        temperature=0.1,
        max_output_tokens=120,
        prompt_version=version,
        response_mime_type="application/json",
    )
    if not response["success"]:
        note = "⏳ Gemini API Free Tier rate limit reached. Please wait 30 seconds." if response["error"] in {"rate_limited", "local_rate_limited"} else "Gemini attribution is unavailable right now."
        return {"family": "Error", "specific_model": "Error", "confidence": 0.0, "note": note}
    try:
        data = json.loads(response["text"])
        family, specific_model = data.get("family"), data.get("specific_model")
        confidence, note = float(data.get("confidence")), data.get("note")
        if family not in _FAMILIES or specific_model not in _MODELS or not isinstance(note, str):
            raise ValueError("invalid response schema")
        return {"family": family, "specific_model": specific_model, "confidence": min(1.0, max(0.0, confidence)), "note": note[:300]}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"family": "Unknown", "specific_model": "Unknown", "confidence": 0.0, "note": "Gemini attribution response could not be validated."}
