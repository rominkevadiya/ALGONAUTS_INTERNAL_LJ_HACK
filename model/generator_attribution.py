from typing import Any, Dict
import warnings

from PIL import Image

from app.api.gemini_gateway import generate_multimodal

warnings.filterwarnings("ignore", message=".*Direct use of automatic function calling.*")


def predict_generator_attribution(image: Image.Image) -> Dict[str, Any]:
    """Predict the likely generator family using the centralized Gemini gateway."""
    prompt = """
    This image has been identified as an AI-generated (FAKE) image. 
    Your task is to analyze its visual artifacts and identify the likely generator family.
    Common families include:
    - Diffusion Models (e.g., Stable Diffusion, Midjourney, DALL-E)
    - GANs (e.g., StyleGAN)
    - Autoregressive / Other

    Look for specific artifacts: GANs often have strange background distortions or asymmetric features (e.g. mismatched earrings). Diffusion models sometimes struggle with complex overlapping structures, text, and hands, but have highly coherent global lighting.

    Output your response in exactly this format:
    FAMILY: [Diffusion / GAN / Unknown]
    SPECIFIC_MODEL: [Midjourney / Stable Diffusion / DALL-E / StyleGAN / Unknown]
    CONFIDENCE: [0.0 to 1.0]
    NOTE: [Brief explanation of why you chose this family]
    """
    response = generate_multimodal(task_id="generator_attribution", image=image, prompt=prompt, temperature=0.1, max_output_tokens=150)
    if not response["success"]:
        note = "⏳ Gemini API Free Tier rate limit reached. Please wait 30 seconds." if response["error"] == "rate_limited" else "Gemini attribution is unavailable right now."
        return {"family": "Error", "specific_model": "Error", "confidence": 0.0, "note": note}

    family, specific_model, confidence, note = "Unknown", "Unknown", 0.0, "N/A"
    for line in response["text"].split("\n"):
        line = line.strip()
        if line.startswith("FAMILY:"):
            family = line.replace("FAMILY:", "").strip()
        elif line.startswith("SPECIFIC_MODEL:"):
            specific_model = line.replace("SPECIFIC_MODEL:", "").strip()
        elif line.startswith("CONFIDENCE:"):
            try:
                confidence = float(line.replace("CONFIDENCE:", "").strip())
            except ValueError:
                pass
        elif line.startswith("NOTE:"):
            note = line.replace("NOTE:", "").strip()
    return {"family": family, "specific_model": specific_model, "confidence": confidence, "note": note}
