"""
SignalScope Predictor API Facade
Re-exports strategy and diagnostic components from modular packages for backward compatibility.
"""

from app.diagnostics.entropy import (
    interpret_confidence,
    compute_prediction_entropy,
)
import time
import logging

# Configure terminal logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

from typing import Dict, Any
from app.config import PATCH_N

from app.diagnostics.disagreement import (
    compute_prediction_disagreement,
)
from app.diagnostics.fft_spectral import (
    compute_fft_spectral_diagnostic,
)
from app.diagnostics.metadata_inspector import (
    inspect_image_metadata,
)
from app.strategies.base_strategy import (
    BaseStrategy,
    validate_strategy_output,
)
from app.strategies.strategy_registry import (
    StrategyRegistry,
    get_strategy,
    list_strategies,
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
from app.strategies.multiscale.multiscale_strategy import (
    predict_image_multiscale,
)
from app.strategies.auto.auto_strategy import (
    predict_image_auto as _predict_image_auto,
    predict_batch,
)

def predict_image_auto(
    image,
    model=None,
    device=None,
    mode: str = "auto",
    n_patches: int = PATCH_N,
    seed: int = 42,
    aggregation: str = "mean",
    precomputed_metadata: Dict[str, Any] = None,
    raw_bytes=None,
):
    """
    Unified automatic dispatcher supporting modes: 'auto', 'multiscale', 'resize', 'patch', 'hybrid', 'tta'.

    Args:
        raw_bytes: Original file bytes from the upload. Passed to the metadata
                   inspector for C2PA JUMBF binary scanning. Without this, PIL
                   re-encoding strips the manifest from the byte stream.
    """
    logger.info(f"🚀 Starting Inference | Requested Mode: {mode.upper()}")
    start_time = time.time()
    
    result = _predict_image_auto(
        image=image,
        model=model,
        device=device,
        mode=mode,
        n_patches=n_patches,
        seed=seed,
        aggregation=aggregation,
        precomputed_metadata=precomputed_metadata,
        raw_bytes=raw_bytes,
    )
    
    logger.info(f"🏁 Final Result: {result}")
    return result

__all__ = [
    "interpret_confidence",
    "compute_prediction_entropy",
    "compute_prediction_disagreement",
    "compute_fft_spectral_diagnostic",
    "inspect_image_metadata",
    "BaseStrategy",
    "validate_strategy_output",
    "StrategyRegistry",
    "get_strategy",
    "list_strategies",
    "prepare_image",
    "get_inference_transform",
    "get_patch_transform",
    "preprocess_image",
    "extract_native_patches",
    "predict_image",
    "predict_image_patch_vote",
    "predict_image_tta",
    "predict_image_hybrid",
    "predict_image_multiscale",
    "predict_image_auto",
    "predict_batch",
]
