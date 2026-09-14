from typing import Any, Dict
import warnings

from PIL import Image

from app.api.gemini_gateway import generate_multimodal

warnings.filterwarnings("ignore", message=".*Direct use of automatic function calling.*")


def generate_faithful_explanation(image: Image.Image, prediction_label: str, regions: list = None, caption: str = None, diagnostic_context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Generate the existing explanation and optional consistency result via Gemini."""
    prompt = f"This image has been classified by an AI detection system as '{prediction_label}' (either REAL or FAKE AI-generated)."
    prompt += "\nYour task is to provide a very brief, faithful explanation for this verdict based on visual cues. Focus on lighting, physical inconsistencies, warped textures, or anatomical errors if it's FAKE, or natural physical consistency if REAL."
    prompt += "\nCRITICAL: Do not fabricate visual cues you cannot clearly see. Only cite artifacts that are genuinely present in the image. If you are uncertain about a specific cue, do not include it. Your explanation must be grounded in the actual image."
    if diagnostic_context:
        prompt += "\n\nIncorporate the following algorithmic diagnostics into your explanation to prove it is mathematically grounded:"
        if "entropy" in diagnostic_context:
            prompt += f"\n- Shannon Entropy: {diagnostic_context.get('normalized_entropy', 0.0):.4f} (Indicates model uncertainty. High entropy = near decision boundary)."
        if "fft_diagnostic" in diagnostic_context:
            fft = diagnostic_context["fft_diagnostic"]
            prompt += f"\n- 2D FFT Spectral Diagnostic: High/Low Energy Ratio is {fft.get('high_to_low_ratio', 0.0):.4f}. {fft.get('interpretation', '')}"
        if "stability" in diagnostic_context:
            stab = diagnostic_context["stability"]
            prompt += f"\n- Native Patch Stability: {stab.get('patch_agreement_pct', 0.0):.1f}% agreement across image patches."
    if caption:
        prompt += f"\n\nAdditionally, a user provided the following caption/claim: '{caption}'."
        prompt += "\nAssess the consistency between the image and this caption. Does the text match the image content?"
        prompt += "\nOutput your response in exactly this format:\nEXPLANATION: [your explanation]\nCONSISTENCY_SCORE: [0.0 to 1.0, where 1.0 is perfectly consistent]\nCONSISTENCY_NOTE: [brief note on consistency]"
    else:
        prompt += "\n\nOutput your response in exactly this format:\nEXPLANATION: [your explanation]"

    response = generate_multimodal(task_id="faithful_explanation", image=image, prompt=prompt, temperature=0.2, max_output_tokens=300)
    if not response["success"]:
        explanation = "⏳ Gemini API Free Tier rate limit reached (15 requests/minute). Please wait 30 seconds and try again!" if response["error"] == "rate_limited" else "Failed to generate explanation via Gemini API."
        return {"explanation": explanation, "consistency_score": None, "consistency_note": "N/A"}

    explanation, consistency_score, consistency_note = "", None, "N/A"
    for line in response["text"].split("\n"):
        line = line.strip()
        if line.startswith("EXPLANATION:"):
            explanation = line.replace("EXPLANATION:", "").strip()
        elif line.startswith("CONSISTENCY_SCORE:"):
            try:
                consistency_score = float(line.replace("CONSISTENCY_SCORE:", "").strip())
            except ValueError:
                pass
        elif line.startswith("CONSISTENCY_NOTE:"):
            consistency_note = line.replace("CONSISTENCY_NOTE:", "").strip()
    if not explanation:
        explanation = response["text"]
    return {"explanation": explanation, "consistency_score": consistency_score, "consistency_note": consistency_note}
