"""Versioned, compact Gemini prompt templates for SignalScope tasks."""

from __future__ import annotations

import json
from typing import Any, Dict, Tuple

GENERATOR_ATTRIBUTION_V1 = "generator_attribution_v1"
FAITHFUL_EXPLANATION_V1 = "faithful_explanation_v1"
MULTIMODAL_CONSISTENCY_V1 = "multimodal_consistency_v1"


def generator_attribution_prompt() -> Tuple[str, str]:
    return GENERATOR_ATTRIBUTION_V1, (
        "Inspect this image only as a likely AI-generated image. Identify visual evidence for its likely generator family; do not claim certainty. "
        "Return compact JSON only: {\"family\":\"Diffusion|GAN|Unknown\",\"specific_model\":\"Midjourney|Stable Diffusion|DALL-E|StyleGAN|Unknown\",\"confidence\":0.0,\"note\":\"short evidence-based reason\"}. "
        "Use Unknown when unsupported."
    )


def faithful_explanation_prompt(prediction_label: str, diagnostic_context: Dict[str, Any] | None, caption: str | None = None) -> Tuple[str, str]:
    diagnostics: Dict[str, Any] = {}
    if diagnostic_context:
        if "entropy" in diagnostic_context:
            diagnostics["normalized_entropy"] = round(float(diagnostic_context.get("normalized_entropy", 0.0)), 4)
        if "fft_diagnostic" in diagnostic_context:
            diagnostics["fft_high_low_ratio"] = round(float(diagnostic_context["fft_diagnostic"].get("high_to_low_ratio", 0.0)), 4)
        if "stability" in diagnostic_context:
            diagnostics["patch_agreement_pct"] = round(float(diagnostic_context["stability"].get("patch_agreement_pct", 0.0)), 1)
    context = {"resnet_verdict": prediction_label, "diagnostics": diagnostics}
    if caption:
        context["caption_claim"] = caption
    prompt = (
        "Explain the supplied SignalScope verdict using only clearly visible image evidence and this compact context: "
        f"{json.dumps(context, separators=(',', ':'))}. "
        "The local ResNet verdict is primary; do not infer identity, political events, or unseen facts. "
        "Return compact JSON only: {\"explanation\":\"one short grounded explanation\",\"consistency_score\":null,\"consistency_note\":\"N/A\"}. "
    )
    if caption:
        prompt += "For caption_claim, set consistency_score from 0 to 1 and give a brief consistency_note."
    return FAITHFUL_EXPLANATION_V1, prompt


def multimodal_consistency_prompt(caption: str) -> Tuple[str, str]:
    return MULTIMODAL_CONSISTENCY_V1, (
        "Compare this image with the supplied caption claim. Report only visible image-caption consistency; do not identify people, infer political events, or judge whether the image is AI-generated. "
        f"Caption claim: {json.dumps(caption)}. "
        "Return compact JSON only: {\"consistent\":true,\"confidence\":0.0,\"note\":\"short visible-evidence note\"}."
    )
