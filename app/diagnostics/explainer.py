import os
from typing import Dict, Any, Optional
from PIL import Image
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore", message=".*Direct use of automatic function calling.*")
load_dotenv()

# We try to import the new google-genai library
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


def generate_faithful_explanation(image: Image.Image, prediction_label: str, regions: list = None, caption: str = None, diagnostic_context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Generates a faithful explanation and (optionally) checks multimodal consistency.
    Uses the Gemini API if GEMINI_API_KEY is available in the environment.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not GENAI_AVAILABLE or not api_key:
        return {
            "explanation": "Gemini API key not found or google-genai not installed. Explanation not available.",
            "consistency_score": None,
            "consistency_note": "N/A"
        }

    client = genai.Client(api_key=api_key)

    prompt = f"This image has been classified by an AI detection system as '{prediction_label}' (either REAL or FAKE AI-generated)."
    prompt += "\nYour task is to provide a very brief, faithful explanation for this verdict based on visual cues. Focus on lighting, physical inconsistencies, warped textures, or anatomical errors if it's FAKE, or natural physical consistency if REAL."
    prompt += "\nCRITICAL: Do not fabricate visual cues you cannot clearly see. Only cite artifacts that are genuinely present in the image. If you are uncertain about a specific cue, do not include it. Your explanation must be grounded in the actual image."
    
    if diagnostic_context:
        prompt += "\n\nIncorporate the following algorithmic diagnostics into your explanation to prove it is mathematically grounded:"
        if 'entropy' in diagnostic_context:
            prompt += f"\n- Shannon Entropy: {diagnostic_context.get('normalized_entropy', 0.0):.4f} (Indicates model uncertainty. High entropy = near decision boundary)."
        if 'fft_diagnostic' in diagnostic_context:
            fft = diagnostic_context['fft_diagnostic']
            prompt += f"\n- 2D FFT Spectral Diagnostic: High/Low Energy Ratio is {fft.get('high_to_low_ratio', 0.0):.4f}. {fft.get('interpretation', '')}"
        if 'stability' in diagnostic_context:
            stab = diagnostic_context['stability']
            prompt += f"\n- Native Patch Stability: {stab.get('patch_agreement_pct', 0.0):.1f}% agreement across image patches."
            
    if caption:
        prompt += f"\n\nAdditionally, a user provided the following caption/claim: '{caption}'."
        prompt += "\nAssess the consistency between the image and this caption. Does the text match the image content?"
        prompt += "\nOutput your response in exactly this format:\nEXPLANATION: [your explanation]\nCONSISTENCY_SCORE: [0.0 to 1.0, where 1.0 is perfectly consistent]\nCONSISTENCY_NOTE: [brief note on consistency]"
    else:
        prompt += "\n\nOutput your response in exactly this format:\nEXPLANATION: [your explanation]"

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[image, prompt],
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=300,
            )
        )
        
        text = response.text
        
        explanation = ""
        consistency_score = None
        consistency_note = "N/A"
        
        for line in text.split('\n'):
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
                
        # Fallback if parsing fails
        if not explanation:
            explanation = text

        return {
            "explanation": explanation,
            "consistency_score": consistency_score,
            "consistency_note": consistency_note
        }

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            explanation = "⏳ Gemini API Free Tier rate limit reached (15 requests/minute). Please wait 30 seconds and try again!"
        else:
            explanation = f"Failed to generate explanation via Gemini API: {error_msg}"
            
        return {
            "explanation": explanation,
            "consistency_score": None,
            "consistency_note": "N/A"
        }
