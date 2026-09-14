import logging
from typing import Dict, Any, List
from PIL import Image
import torch
import pandas as pd

logger = logging.getLogger(__name__)

from app.config import (
    DEFAULT_INFERENCE_MODE,
    PATCH_AGGREGATION_DEFAULT,
    METADATA_OVERRIDE_CONFIDENCE_CAP,
)
from app.model_loader import load_model, resolve_model_device
from app.diagnostics.entropy import compute_prediction_entropy
from app.diagnostics.disagreement import compute_prediction_disagreement
from app.diagnostics.fft_spectral import compute_fft_spectral_diagnostic
from app.strategies.patch.patch_extractor import prepare_image
from app.strategies.resize.resize_strategy import predict_image
from app.strategies.patch.patch_strategy import predict_image_patch_vote
from app.strategies.tta.tta_strategy import predict_image_tta
from app.strategies.hybrid.hybrid_strategy import predict_image_hybrid


from app.strategies.base_strategy import BaseStrategy, validate_strategy_output
from app.strategies.strategy_registry import register_strategy


def predict_image_auto(
    image: Image.Image,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    mode: str = DEFAULT_INFERENCE_MODE,
    n_patches: int = 0,  # 0 enables dynamic patch count based on resolution
    seed: int = 42,
    aggregation: str = PATCH_AGGREGATION_DEFAULT,
    precomputed_metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Unified automatic dispatcher supporting modes: 'auto', 'multiscale', 'resize', 'patch', 'hybrid', 'tta'.
    """
    clean_img = prepare_image(image)
    w, h = clean_img.size
    min_dim = min(w, h)
    max_dim = max(w, h)
    
    logger.info(f"Image Resolution: {w}x{h}")

    # Resolution & context-aware automatic selection logic
    selected_mode = mode
    if mode == "auto":
        if min_dim < 64:
            selected_mode = "resize"
        elif min_dim < 256:
            selected_mode = "patch"
        else:
            selected_mode = "hybrid"


    # Compute FFT once at the dispatcher level for hybrid/auto to avoid double computation
    logger.info("Computing 2D FFT Spectral Diagnostic...")
    fft_diagnostic = compute_fft_spectral_diagnostic(clean_img)

    # Use precomputed metadata if provided (preserves C2PA manifest from raw_bytes), otherwise compute it
    logger.info("Checking Metadata & C2PA Provenance...")
    if precomputed_metadata is not None:
        meta_diagnostic = precomputed_metadata
    else:
        from app.diagnostics.metadata_inspector import inspect_image_metadata
        meta_diagnostic = inspect_image_metadata(clean_img)

    has_metadata_override = False
    source_id = ""
    # Detect if verified AI metadata or C2PA manifest is present, but don't return early
    if meta_diagnostic["provenance_verdict"] == "AI_GENERATED":
        has_metadata_override = True
        source_id = meta_diagnostic.get("source_identified") or "AI Generator Provenance Tag"
        logger.info(f"C2PA / Metadata Match found: {source_id}. Will override final confidence after running PyTorch model.")

    logger.info(f"Executing Deep Learning Model in Mode: {selected_mode.upper()}")
    if selected_mode == "resize":
        result = predict_image(clean_img, model=model, device=device)
    elif selected_mode == "patch":
        result = predict_image_patch_vote(
            clean_img, model=model, device=device, n_patches=n_patches, seed=seed, aggregation=aggregation
        )
    elif selected_mode == "hybrid":
        result = predict_image_hybrid(
            clean_img, model=model, device=device, n_patches=n_patches, seed=seed, aggregation=aggregation,
            _precomputed_fft=fft_diagnostic
        )
    elif selected_mode == "multiscale":
        from app.strategies.multiscale.multiscale_strategy import predict_image_multiscale
        result = predict_image_multiscale(clean_img, model=model, device=device)
    elif selected_mode == "tta":
        result = predict_image_tta(clean_img, model=model, device=device)
    else:
        # Fallback to multiscale
        from app.strategies.multiscale.multiscale_strategy import predict_image_multiscale
        result = predict_image_multiscale(clean_img, model=model, device=device)

    # Attach entropy, disagreement, FFT, and metadata diagnostics to output dictionary
    entropy_info = compute_prediction_entropy(result["fake_probability"], result["real_probability"])
    result.update(entropy_info)

    if "patch_fake_probs" in result:
        disagreement_info = compute_prediction_disagreement(result["patch_fake_probs"])
        result["stability"] = disagreement_info

    result["fft_diagnostic"] = fft_diagnostic
    result["metadata_diagnostic"] = meta_diagnostic
    result["image_dimensions"] = f"{w} x {h}"
    
    if has_metadata_override:
        result["label"] = "FAKE"
        result["confidence"] = METADATA_OVERRIDE_CONFIDENCE_CAP
        result["inference_mode"] = "metadata_provenance"
        result["agreement"] = f"C2PA / AI Provenance Match ({source_id}) combined with {selected_mode.upper()}"
        result["uncertainty_level"] = "High Confidence (Metadata Match)"
        result["uncertainty_note"] = f"Image contains AI-related metadata: {source_id}. PyTorch model was run to extract features, but final verdict was overridden by C2PA metadata."
        
    return validate_strategy_output(result, strategy_name="auto")


class AutoStrategy(BaseStrategy):
    """Concrete BaseStrategy implementation for Resolution & Metadata Aware Auto Dispatcher."""

    @property
    def name(self) -> str:
        return "auto"

    @property
    def display_name(self) -> str:
        return "Auto (Smart Pipeline Router)"

    @property
    def description(self) -> str:
        return "Automatically checks C2PA/EXIF metadata, then routes image to best strategy based on resolution."

    def predict(
        self,
        image: Image.Image,
        model: torch.nn.Module | None = None,
        device: torch.device | None = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        return predict_image_auto(image, model=model, device=device, **kwargs)


# Register strategy with StrategyRegistry
register_strategy(AutoStrategy())


def predict_batch(
    images_dict: Dict[str, Image.Image],
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    mode: str = DEFAULT_INFERENCE_MODE,
    n_patches: int = 0,
    aggregation: str = PATCH_AGGREGATION_DEFAULT
) -> pd.DataFrame:
    """
    Processes a dictionary of {filename: PIL.Image} and returns a structured pandas DataFrame.
    """
    model, device = resolve_model_device(model, device)

    results: List[Dict[str, Any]] = []

    for idx, (filename, img) in enumerate(images_dict.items()):
        try:
            res = predict_image_auto(
                img, model=model, device=device, mode=mode, n_patches=n_patches, aggregation=aggregation
            )
            results.append({
                "Filename": filename,
                "Prediction": res["label"],
                "Confidence": f"{res['confidence'] * 100:.2f}%",
                "Fake Probability": f"{res['fake_probability'] * 100:.2f}%",
                "Real Probability": f"{res['real_probability'] * 100:.2f}%",
                "Inference Mode": res.get("inference_mode", mode),
                "Uncertainty": res.get("uncertainty_level", "N/A"),
                "Raw Confidence": res["confidence"]
            })
        except Exception as e:
            logging.error(f"Error processing {filename}: {e}", exc_info=True)
            results.append({
                "Filename": filename,
                "Prediction": "ERROR",
                "Confidence": "N/A",
                "Fake Probability": "N/A",
                "Real Probability": "N/A",
                "Inference Mode": mode,
                "Uncertainty": "ERROR",
                "Raw Confidence": 0.0
            })

    return pd.DataFrame(results)