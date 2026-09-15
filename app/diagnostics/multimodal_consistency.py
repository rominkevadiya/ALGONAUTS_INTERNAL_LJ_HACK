"""Optional image-caption consistency helper (SignalScope Module E)."""

from __future__ import annotations

import json
import re
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
        raw_text = response["text"].strip()
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
            raw_text = re.sub(r"\s*```$", "", raw_text).strip()
        data = json.loads(raw_text)
        consistent = bool(data.get("consistent", False))
        confidence = float(data.get("confidence", 0.0))
        note = str(data.get("note", "N/A"))
        return {"consistent": consistent, "confidence": min(1.0, max(0.0, confidence)), "note": note[:300]}
    except Exception:
        return {"consistent": False, "confidence": 0.0, "note": "Image-caption consistency response could not be validated."}
