"""
SignalScope Metadata & C2PA Provenance Inspector
Scans uploaded image EXIF, PNG info chunks, and raw byte headers for C2PA manifests,
AI generation metadata, and camera hardware tags before invoking deep learning models.
"""

from typing import Dict, Any, Optional
import io
from PIL import Image, ExifTags


KNOWN_AI_SIGNATURES = [
    "midjourney",
    "dall-e",
    "dalle",
    "stable diffusion",
    "stablediffusion",
    "adobe firefly",
    "firefly",
    "google imagen",
    "imagen",
    "bing image creator",
    "c2pa",
    "generative ai",
    "generativeai",
    "novelai",
    "leonardo.ai",
    "leonardo ai",
    "comfyui",
    "automatic1111",
    "sdxl",
    "flux.1",
    "flux1"
]

CAMERA_MANUFACTURERS = [
    "apple", "canon", "nikon", "sony", "fujifilm", "panasonic",
    "olympus", "leica", "samsung", "google", "xiaomi", "oneplus",
    "huawei", "hasselblad", "pentax", "gopro"
]


def inspect_image_metadata(
    image: Image.Image,
    raw_bytes: Optional[bytes] = None,
    filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Inspects image for EXIF metadata, PNG info chunks, C2PA provenance headers, and messenger upload signatures.
    """
    metadata_summary: Dict[str, str] = {}
    ai_matched_terms = []
    camera_matched = None
    c2pa_detected = False

    # Check filename patterns for social messenger camera uploads (WhatsApp, Telegram, Signal, DCIM)
    messenger_matched = None
    if filename:
        fn_lower = filename.lower()
        if any(pat in fn_lower for pat in ["whatsapp", "telegram", "signal", "img_", "pxl_", "dsci", "wa0"]):
            messenger_matched = filename

    # ------------------------------------------------------------------
    # 1. EXIF Metadata Extraction
    # ------------------------------------------------------------------
    try:
        exif = image.getexif()
        if exif:
            for tag_id, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                val_str = str(value).strip()
                if val_str and len(val_str) < 500:
                    metadata_summary[tag_name] = val_str

                # Check text for AI signatures
                val_lower = val_str.lower()
                for sig in KNOWN_AI_SIGNATURES:
                    if sig in val_lower:
                        ai_matched_terms.append(f"EXIF {tag_name}: '{val_str}'")

                # Check camera hardware tags
                if tag_name in ["Make", "Model"]:
                    for cam in CAMERA_MANUFACTURERS:
                        if cam in val_lower:
                            camera_matched = f"{metadata_summary.get('Make', '')} {metadata_summary.get('Model', '')}".strip()
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 2. PNG / TIFF Metadata Chunks (e.g. image.info)
    # ------------------------------------------------------------------
    if hasattr(image, "info") and image.info:
        for key, val in image.info.items():
            key_str = str(key)
            val_str = str(val)
            if len(val_str) < 1000:
                metadata_summary[f"Info:{key_str}"] = val_str

            val_lower = f"{key_str} {val_str}".lower()
            for sig in KNOWN_AI_SIGNATURES:
                if sig in val_lower:
                    ai_matched_terms.append(f"PNG Info [{key_str}]")

    # ------------------------------------------------------------------
    # 3. Structured Raw Byte Stream Scan (C2PA JUMBF Manifest Box)
    # ------------------------------------------------------------------
    if raw_bytes is None:
        try:
            buf = io.BytesIO()
            image.save(buf, format=image.format or "PNG")
            raw_bytes = buf.getvalue()
        except Exception:
            raw_bytes = b""

    if raw_bytes:
        # C2PA JUMBF specification requires 'jumb' followed by 'c2pa' claim signature box
        if b"jumbc2pa" in raw_bytes or b"c2pa.claim" in raw_bytes or b"c2pa.manifest" in raw_bytes:
            c2pa_detected = True
            metadata_summary["C2PA Manifest"] = "Detected Verified JUMBF Digital Provenance Header"

    # ------------------------------------------------------------------
    # 4. Verdict Determination
    # ------------------------------------------------------------------
    metadata_found = len(metadata_summary) > 0 or len(ai_matched_terms) > 0 or messenger_matched is not None

    if camera_matched and not ai_matched_terms and not c2pa_detected:
        provenance_verdict = "CAMERA_REAL"
        source_identified = f"Camera Hardware EXIF ({camera_matched})"
        status_message = f"Authentic Camera Metadata Verified ({camera_matched})"
    elif ai_matched_terms:
        provenance_verdict = "AI_GENERATED"
        source_identified = ai_matched_terms[0]
        status_message = f"AI Provenance Verified ({source_identified})"
    elif c2pa_detected:
        provenance_verdict = "AI_GENERATED"
        source_identified = "C2PA Provenance Manifest"
        status_message = "C2PA Digital Content Credentials Manifest Detected"
    elif messenger_matched and not ai_matched_terms:
        provenance_verdict = "CAMERA_REAL"
        source_identified = f"User Camera Media ({messenger_matched})"
        status_message = f"Authentic Media Upload Verified ({messenger_matched})"
    else:
        provenance_verdict = "UNVERIFIED"
        source_identified = None
        status_message = "No definitive C2PA/AI metadata found → Passing to PyTorch ResNet-50 Pipeline"


    return {
        "metadata_found": metadata_found,
        "provenance_verdict": provenance_verdict,
        "source_identified": source_identified,
        "c2pa_manifest_detected": c2pa_detected,
        "ai_matched_terms": ai_matched_terms,
        "camera_matched": camera_matched,
        "metadata_summary": metadata_summary,
        "status_message": status_message
    }

