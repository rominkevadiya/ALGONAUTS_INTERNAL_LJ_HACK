"""
SignalScope Predictor API Facade
Re-exports strategy and diagnostic components from modular packages for backward compatibility.
"""

from app.diagnostics.entropy import (
    interpret_confidence,
    compute_prediction_entropy,
)
from app.diagnostics.disagreement import (
    compute_prediction_disagreement,
)
from app.diagnostics.fft_spectral import (
    compute_fft_spectral_diagnostic,
)
from app.diagnostics.metadata_inspector import (
    inspect_image_metadata,
)
from app.strategies.patch.patch_extractor import (
    prepare_image,
    get_inference_transform,
    get_patch_transform,
    preprocess_image,
    extract_native_patches,
)
from app.strategies.resize.resize_strategy import (
    predict_image,
)
from app.strategies.patch.patch_strategy import (
    predict_image_patch_vote,
)
from app.strategies.tta.tta_strategy import (
    predict_image_tta,
)
from app.strategies.hybrid.hybrid_strategy import (
    predict_image_hybrid,
)
from app.strategies.auto.auto_strategy import (
    predict_image_auto,
    predict_batch,
)

__all__ = [
    "interpret_confidence",
    "compute_prediction_entropy",
    "compute_prediction_disagreement",
    "compute_fft_spectral_diagnostic",
    "inspect_image_metadata",
    "prepare_image",
    "get_inference_transform",
    "get_patch_transform",
    "preprocess_image",
    "extract_native_patches",
    "predict_image",
    "predict_image_patch_vote",
    "predict_image_tta",
    "predict_image_hybrid",
    "predict_image_auto",
    "predict_batch",
]
