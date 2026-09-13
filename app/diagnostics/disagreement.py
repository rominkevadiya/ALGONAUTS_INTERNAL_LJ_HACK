from typing import List, Dict, Any
import numpy as np


def compute_prediction_disagreement(probabilities: List[float]) -> Dict[str, Any]:
    """
    Calculates statistical disagreement metrics over a set of patch or TTA probabilities.
    """
    if not probabilities:
        return {
            "mean_fake_probability": 0.0,
            "median_fake_probability": 0.0,
            "std_fake_probability": 0.0,
            "min_fake_probability": 0.0,
            "max_fake_probability": 0.0,
            "range_fake_probability": 0.0,
            "fake_patch_count": 0,
            "real_patch_count": 0,
            "patch_agreement_pct": 100.0
        }

    probs = [float(p) for p in probabilities]
    n = len(probs)
    mean_p = float(np.mean(probs))
    med_p = float(np.median(probs))
    std_p = float(np.std(probs))
    min_p = float(np.min(probs))
    max_p = float(np.max(probs))
    range_p = max_p - min_p

    fake_count = sum(1 for p in probs if p > 0.5)
    real_count = n - fake_count
    majority_count = max(fake_count, real_count)
    agreement_pct = (majority_count / n) * 100.0

    return {
        "mean_fake_probability": mean_p,
        "median_fake_probability": med_p,
        "std_fake_probability": std_p,
        "min_fake_probability": min_p,
        "max_fake_probability": max_p,
        "range_fake_probability": range_p,
        "fake_patch_count": fake_count,
        "real_patch_count": real_count,
        "patch_agreement_pct": float(agreement_pct)
    }
