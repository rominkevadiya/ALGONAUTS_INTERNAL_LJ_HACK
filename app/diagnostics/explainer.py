import json
import re
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
        raw_text = response["text"].strip()
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
            raw_text = re.sub(r"\s*```$", "", raw_text).strip()
        data = json.loads(raw_text)
        explanation = data.get("explanation")
        note = data.get("consistency_note", "N/A")
        if not isinstance(explanation, str):
            raise ValueError("invalid response schema")
        note = str(note) if note is not None else "N/A"
        raw_score = data.get("consistency_score")
        consistency_score = None
        if raw_score is not None:
            try:
                val = float(raw_score)
                if val > 1.0 and val <= 10.0:
                    val = val / 10.0
                elif val > 10.0:
                    val = val / 100.0
                consistency_score = min(1.0, max(0.0, val))
            except (ValueError, TypeError):
                consistency_score = None
        return {"explanation": explanation[:600], "consistency_score": consistency_score, "consistency_note": note[:300]}
    except Exception:
        return {"explanation": "Gemini explanation response could not be validated.", "consistency_score": None, "consistency_note": "N/A"}
