import math
from typing import Dict, Any
from app.config import (
    HIGH_CONFIDENCE_THRESHOLD,
    MODERATE_CONFIDENCE_THRESHOLD,
    ENTROPY_LOW_THRESHOLD,
    ENTROPY_HIGH_THRESHOLD,
)


def interpret_confidence(confidence: float) -> Dict[str, str]:
    """
    Categorizes model confidence score into intuitive levels for UI display.
    """
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return {
            "level": "High Confidence",
            "status": "success",
            "message": "Model is highly confident in this prediction."
        }
    elif confidence >= MODERATE_CONFIDENCE_THRESHOLD:
        return {
            "level": "Moderate Confidence",
            "status": "info",
            "message": "Model prediction has moderate certainty."
        }
    else:
        return {
            "level": "Low Confidence / Review Recommended",
            "status": "warning",
            "message": "Prediction score is near the decision boundary. Verification recommended."
        }


def compute_prediction_entropy(fake_prob: float, real_prob: float) -> Dict[str, Any]:
    """
    Computes normalized binary Shannon entropy:
    H(p) = -p log2(p) - (1-p) log2(1-p)
    Normalized H_norm = H(p) / log2(2) = H(p)
    """
    p_fake = max(1e-9, min(1.0 - 1e-9, float(fake_prob)))
    p_real = max(1e-9, min(1.0 - 1e-9, float(real_prob)))

    h = -(p_fake * math.log2(p_fake) + p_real * math.log2(p_real))
    norm_h = float(min(1.0, max(0.0, h)))

    if norm_h < ENTROPY_LOW_THRESHOLD:
        level = "Low Output Entropy"
        note = "Model prediction is highly decisive. Output probability distribution is concentrated."
    elif norm_h < ENTROPY_HIGH_THRESHOLD:
        level = "Moderate Output Entropy"
        note = "Model prediction shows moderate uncertainty."
    else:
        level = "High Output Entropy"
        note = "Model prediction is near the 50/50 decision boundary. High output uncertainty."

    return {
        "entropy": norm_h * math.log(2),  # Natural log entropy
        "normalized_entropy": norm_h,
        "uncertainty_level": level,
        "uncertainty_note": note + " (Note: Entropy measures model output certainty, not guaranteed correctness)."
    }
