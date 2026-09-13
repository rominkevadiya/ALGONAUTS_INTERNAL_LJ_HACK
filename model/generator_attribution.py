import os
from typing import Dict, Any
from PIL import Image
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore", message=".*Direct use of automatic function calling.*")
load_dotenv()

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


def predict_generator_attribution(image: Image.Image) -> Dict[str, Any]:
    """
    Predicts the likely generator family (GAN, Diffusion, Specific Model) of an AI-generated image
    using the Gemini Vision API.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not GENAI_AVAILABLE or not api_key:
        return {
            "family": "Unknown",
            "confidence": 0.0,
            "note": "Gemini API key not found or google-genai not installed."
        }

    client = genai.Client(api_key=api_key)

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

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[image, prompt],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=150,
            )
        )
        
        text = response.text
        
        family = "Unknown"
        specific_model = "Unknown"
        confidence = 0.0
        note = "N/A"
        
        for line in text.split('\n'):
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
                
        return {
            "family": family,
            "specific_model": specific_model,
            "confidence": confidence,
            "note": note
        }

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            note = "⏳ Gemini API Free Tier rate limit reached. Please wait 30 seconds."
        else:
            note = f"API Error: {error_msg}"
            
        return {
            "family": "Error",
            "specific_model": "Error",
            "confidence": 0.0,
            "note": note
        }
