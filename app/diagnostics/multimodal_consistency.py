"""Optional image-caption consistency helper (SignalScope Module E)."""

from __future__ import annotations

import json
from typing import Any, Dict

from PIL import Image

from app.api.gemini_gateway import generate_multimodal
from app.api.prompt_registry import multimodal_consistency_prompt


def assess_image_caption_consistency(image: Image.Image, caption: str) -> Dict[str, Any]:
    """Assess caption consistency only; this does not classify an image as real or AI."""
    version, prompt = multimodal_consistency_prompt(caption)
    response = generate_multimodal(task_id=version, image=image, prompt=prompt, temperature=0.2, max_output_tokens=100, prompt_version=version, response_mime_type="application/json")
    if not response["success"]:
        return {"consistent": False, "confidence": 0.0, "note": "Image-caption consistency is unavailable right now."}
    try:
        data = json.loads(response["text"])
        consistent, confidence, note = data.get("consistent"), float(data.get("confidence")), data.get("note")
        if not isinstance(consistent, bool) or not isinstance(note, str):
            raise ValueError("invalid response schema")
        return {"consistent": consistent, "confidence": min(1.0, max(0.0, confidence)), "note": note[:300]}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"consistent": False, "confidence": 0.0, "note": "Image-caption consistency response could not be validated."}
