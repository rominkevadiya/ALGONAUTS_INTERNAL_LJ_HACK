"""Versioned, compact Gemini prompt templates for SignalScope tasks."""

from __future__ import annotations

import json
from typing import Any, Dict, Tuple

GENERATOR_ATTRIBUTION_V1 = "generator_attribution_v1"
FAITHFUL_EXPLANATION_V1 = "faithful_explanation_v1"
MULTIMODAL_CONSISTENCY_V1 = "multimodal_consistency_v1"


def generator_attribution_prompt() -> Tuple[str, str]:
    """
    Neutral framing: asks Gemini to determine whether the image is AI-generated
    rather than assuming it is. Includes an expanded list of modern generator families.
    max_output_tokens: 220 (JSON response is compact)
    """
    return GENERATOR_ATTRIBUTION_V1, (
        "Analyze this image's visual characteristics objectively. Determine whether it appears to be "
        "AI-generated (synthetic) or photographically authentic. "
        "If AI-generated, identify the most likely generator family from: "
        "Diffusion (Stable Diffusion, SDXL, Flux.1, Flux.1-dev, Flux.1-schnell), "
        "Commercial Diffusion (Midjourney v4/v5/v6, DALL-E 2/3, Adobe Firefly, Canva AI, Bing Image Creator), "
        "Google (Imagen 2, Imagen 3, ImageFX, Gemini Imagen), "
        "Video/Emerging (Sora, Kling, Runway Gen-2/Gen-3, Pika, Lumiere, Dream Machine), "
        "GAN (StyleGAN, BigGAN), "
        "Unknown AI, or Real Photo. "
        "Base your judgment strictly on visible artifacts: frequency patterns, edge consistency, "
        "texture uniformity, anatomical plausibility, background coherence, and lighting. "
        "Return compact JSON only — no markdown fences: "
        "{\"is_ai_generated\":true,\"family\":\"Diffusion|Commercial Diffusion|Google|GAN|Unknown AI|Real Photo\","
        "\"specific_model\":\"best match or Unknown\",\"confidence\":0.0,"
        "\"note\":\"one-sentence evidence-based reason\"}. "
        "Use Unknown AI when AI-generated but generator is unclear. Use Real Photo when authentic."
    )


def faithful_explanation_prompt(prediction_label: str, diagnostic_context: Dict[str, Any] | None, caption: str | None = None) -> Tuple[str, str]:
    """
    Explains the SignalScope verdict with grounded visual and diagnostic evidence.
    Passes richer diagnostic context including FFT, entropy, patch agreement, and strategy mode.
    max_output_tokens: 300 (up from 180 to prevent mid-sentence truncation in complex cases)
    """
    diagnostics: Dict[str, Any] = {}
    if diagnostic_context:
        if "normalized_entropy" in diagnostic_context:
            diagnostics["normalized_entropy"] = round(float(diagnostic_context.get("normalized_entropy", 0.0)), 4)
        elif "entropy" in diagnostic_context:
            diagnostics["normalized_entropy"] = round(float(diagnostic_context.get("normalized_entropy", 0.0)), 4)
        if "fft_diagnostic" in diagnostic_context:
            fft = diagnostic_context["fft_diagnostic"]
            diagnostics["fft_high_low_ratio"] = round(float(fft.get("high_to_low_ratio", 0.0)), 4)
            diagnostics["fft_spectral_score"] = round(float(fft.get("spectral_score", 0.0)), 4)
        if "stability" in diagnostic_context:
            stab = diagnostic_context["stability"]
            diagnostics["patch_agreement_pct"] = round(float(stab.get("patch_agreement_pct", 0.0)), 1)
            diagnostics["patch_std"] = round(float(stab.get("std_fake_probability", 0.0)), 4)
        if "inference_mode" in diagnostic_context:
            diagnostics["strategy"] = diagnostic_context["inference_mode"]
        if "branch_disagreement" in diagnostic_context:
            diagnostics["branch_disagreement"] = round(float(diagnostic_context.get("branch_disagreement", 0.0)), 4)

    context = {"resnet_verdict": prediction_label, "diagnostics": diagnostics}
    if caption:
        context["caption_claim"] = caption
    prompt = (
        "Explain the supplied SignalScope verdict using only clearly visible image evidence and this compact context: "
        f"{json.dumps(context, separators=(',', ':'))}. "
        "The local ResNet verdict is primary; do not infer identity, political events, or unseen facts. "
        "Return compact JSON only — no markdown fences: "
        "{\"explanation\":\"one to two sentence grounded explanation of visible evidence\","
        "\"consistency_score\":null,\"consistency_note\":\"N/A\"}. "
    )
    if caption:
        prompt += "For caption_claim, set consistency_score from 0 to 1 and give a brief consistency_note."
    return FAITHFUL_EXPLANATION_V1, prompt


def multimodal_consistency_prompt(caption: str) -> Tuple[str, str]:
    return MULTIMODAL_CONSISTENCY_V1, (
        "Compare this image with the supplied caption claim. Report only visible image-caption consistency; "
        "do not identify people, infer political events, or judge whether the image is AI-generated. "
        f"Caption claim: {json.dumps(caption)}. "
        "Return compact JSON only — no markdown fences: "
        "{\"consistent\":true,\"confidence\":0.0,\"note\":\"short visible-evidence note\"}."
    )
