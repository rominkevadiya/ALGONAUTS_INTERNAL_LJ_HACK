import json
from typing import Any, Dict
import warnings

from PIL import Image

from app.api.gemini_gateway import generate_multimodal
from app.api.prompt_registry import faithful_explanation_prompt

warnings.filterwarnings("ignore", message=".*Direct use of automatic function calling.*")


def generate_faithful_explanation(image: Image.Image, prediction_label: str, regions: list = None, caption: str = None, diagnostic_context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Generate the existing concise explanation and optional consistency result via Gemini."""
    version, prompt = faithful_explanation_prompt(prediction_label, diagnostic_context, caption)
    response = generate_multimodal(
        task_id=version,
        image=image,
        prompt=prompt,
        temperature=0.2,
        max_output_tokens=180,
        prompt_version=version,
        response_mime_type="application/json",
    )
    if not response["success"]:
        explanation = "⏳ Gemini API Free Tier rate limit reached (15 requests/minute). Please wait 30 seconds and try again!" if response["error"] in {"rate_limited", "local_rate_limited"} else "Failed to generate explanation via Gemini API."
        return {"explanation": explanation, "consistency_score": None, "consistency_note": "N/A"}
    try:
        data = json.loads(response["text"])
        explanation, score, note = data.get("explanation"), data.get("consistency_score"), data.get("consistency_note", "N/A")
        if not isinstance(explanation, str) or not isinstance(note, str):
            raise ValueError("invalid response schema")
        consistency_score = None if score is None else min(1.0, max(0.0, float(score)))
        return {"explanation": explanation[:600], "consistency_score": consistency_score, "consistency_note": note[:300]}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"explanation": "Gemini explanation response could not be validated.", "consistency_score": None, "consistency_note": "N/A"}
