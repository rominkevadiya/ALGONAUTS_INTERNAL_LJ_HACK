from typing import Dict, Any
from PIL import Image
import numpy as np


def compute_fft_spectral_diagnostic(image: Image.Image) -> Dict[str, Any]:
    """
    Experimental 2D Fast Fourier Transform (FFT) spectral diagnostic.
    Converts image to floating-point grayscale, removes mean intensity, calculates 2D FFT,
    and analyzes radial frequency energy distribution.
    """
    from app.strategies.patch.patch_extractor import prepare_image

    clean_image = prepare_image(image)
    gray = np.array(clean_image.convert("L"), dtype=np.float32)
    
    # Remove mean intensity to eliminate DC spike
    gray_zero_mean = gray - np.mean(gray)
    
    fft = np.fft.fft2(gray_zero_mean)
    shifted = np.fft.fftshift(fft)
    magnitude = np.abs(shifted)
    log_power = np.log1p(magnitude ** 2)

    h, w = log_power.shape
    cy, cx = h // 2, w // 2
    max_radius = min(cy, cx)

    if max_radius < 4:
        return {
            "spectral_score": 0.0,
            "low_frequency_energy": 0.0,
            "high_frequency_energy": 0.0,
            "spectral_slope": 0.0,
            "diagnostic_label": "Low spectral irregularity",
            "interpretation": "Experimental frequency-domain diagnostic only. Affected by image content and resolution."
        }

    y_grid, x_grid = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)

    low_freq_mask = dist_from_center <= (max_radius * 0.25)
    high_freq_mask = (dist_from_center > (max_radius * 0.25)) & (dist_from_center <= max_radius)

    low_energy = float(np.mean(log_power[low_freq_mask])) if np.any(low_freq_mask) else 1e-6
    high_energy = float(np.mean(log_power[high_freq_mask])) if np.any(high_freq_mask) else 1e-6

    ratio = high_energy / (low_energy + 1e-9)
    spectral_score = float(min(1.0, max(0.0, ratio * 3.5)))

    # Radial profile slope estimation
    radii = np.arange(1, max_radius)
    intensity_profile = [float(np.mean(log_power[(dist_from_center >= r-0.5) & (dist_from_center < r+0.5)])) for r in radii]
    valid_mask = ~np.isnan(intensity_profile)
    if np.sum(valid_mask) > 3:
        log_r = np.log(radii[valid_mask])
        log_i = np.array(intensity_profile)[valid_mask]
        slope, _ = np.polyfit(log_r, log_i, 1)
    else:
        slope = 0.0

    if spectral_score < 0.35:
        label = "Low spectral irregularity"
    elif spectral_score < 0.65:
        label = "Moderate spectral irregularity"
    else:
        label = "High spectral irregularity"

    return {
        "spectral_score": spectral_score,
        "low_frequency_energy": low_energy,
        "high_frequency_energy": high_energy,
        "high_to_low_ratio": float(ratio),
        "spectral_slope": float(slope),
        "diagnostic_label": label,
        "interpretation": "Experimental frequency-domain diagnostic only. Not a trained classifier probability."
    }
