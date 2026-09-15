import logging
from typing import Dict, Any, List, Optional
from PIL import Image
import torch
import pandas as pd

logger = logging.getLogger(__name__)

from app.config import (
    DEFAULT_INFERENCE_MODE,
    PATCH_AGGREGATION_DEFAULT,
    METADATA_OVERRIDE_CONFIDENCE_CAP,
    PATCH_N,
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
    n_patches: int = PATCH_N,
    seed: int = 42,
    aggregation: str = PATCH_AGGREGATION_DEFAULT,
    precomputed_metadata: Optional[Dict[str, Any]] = None,
    raw_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
    """
    Unified automatic dispatcher supporting modes: 'auto', 'multiscale', 'resize', 'patch', 'hybrid', 'tta'.

    Args:
        image: Original PIL Image before any preparation steps.
        model: Optional pre-loaded PyTorch model.
        device: Optional torch device.
        mode: Inference strategy key. 'auto' selects based on resolution.
        n_patches: Number of native 32x32 patches. 0 = dynamic.
        seed: Random seed for deterministic patch extraction.
        aggregation: Patch aggregation method.
        precomputed_metadata: Pre-screened metadata dict from app.py upload handler.
                              Reused as-is to avoid re-scanning and to preserve the
                              raw-bytes C2PA result already computed from the original
                              upload file bytes.
        raw_bytes: Original file bytes of the uploaded image. Used for C2PA JUMBF
                   binary scanning when precomputed_metadata is not provided (e.g.
                   batch mode, robustness tests). Without this, PIL re-encoding
                   strips the C2PA manifest from the byte stream.
    """
    clean_img = prepare_image(image)
    w, h = clean_img.size
    min_dim = min(w, h)
    max_dim = max(w, h)
    
    logger.info(f"Image Resolution: {w}x{h}")

    # Resolution & context-aware automatic selection logic
    # Routing tiers (based on minimum dimension):
    #   < 64px   → resize  (tiny image; native patches impossible)
    #   64-255px → patch   (small image; native 32×32 crops cover it)
    #   256-511px → hybrid (medium image; resize + patch + FFT consensus)
    #   ≥ 512px  → multiscale (large/high-res; 3-branch spatial analysis)
    selected_mode = mode
    if mode == "auto":
        if min_dim < 64:
            selected_mode = "resize"
        elif min_dim < 256:
            selected_mode = "patch"
        elif min_dim < 512:
            selected_mode = "hybrid"
        else:
            selected_mode = "multiscale"

    # Compute FFT only when hybrid mode will actually use it (avoids wasted compute for resize/patch/multiscale routes)
    fft_diagnostic = {}
    if selected_mode in ("hybrid",):
        logger.info("Computing 2D FFT Spectral Diagnostic...")
        fft_diagnostic = compute_fft_spectral_diagnostic(clean_img)

    # Use precomputed metadata if provided (preserves C2PA manifest from raw_bytes), otherwise compute it.
    # IMPORTANT: when recomputing, pass the *original* PIL image (not clean_img) and the original
    # raw_bytes so the C2PA JUMBF binary scan can find the manifest. PIL .save() re-encoding
    # strips the JUMBF container, so passing raw_bytes=None causes false-negative C2PA detection.
    logger.info("Checking Metadata & C2PA Provenance...")
    if precomputed_metadata is not None:
        meta_diagnostic = precomputed_metadata
    else:
        from app.diagnostics.metadata_inspector import inspect_image_metadata
        meta_diagnostic = inspect_image_metadata(image, raw_bytes=raw_bytes)

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
        # Preserve the raw model output so the UI can show it for transparency.
        # These keys are optional — downstream code must guard with .get().
        result["model_fake_probability"] = result["fake_probability"]
        result["model_real_probability"] = result["real_probability"]
        result["model_label"] = result["label"]
        # Override all verdict fields to be consistent with the C2PA provenance decision.
        # fake_probability and real_probability MUST match the label or the UI will show
        # a contradictory "Fake: 13% / Real: 87%" while the label says AI-GENERATED.
        result["label"] = "FAKE"
        result["confidence"] = METADATA_OVERRIDE_CONFIDENCE_CAP
        result["fake_probability"] = METADATA_OVERRIDE_CONFIDENCE_CAP
        result["real_probability"] = 1.0 - METADATA_OVERRIDE_CONFIDENCE_CAP
        result["inference_mode"] = "metadata_provenance"
        result["agreement"] = f"C2PA / AI Provenance Match ({source_id}) combined with {selected_mode.upper()}"
        result["uncertainty_level"] = "High Confidence (Metadata Match)"
        result["uncertainty_note"] = (
            f"Image contains verified AI provenance metadata: {source_id}. "
            f"PyTorch model ({selected_mode.upper()}) was executed and its probabilities "
            f"are preserved in model_fake_probability / model_real_probability. "
            f"Final verdict is locked by cryptographic C2PA provenance."
        )
        
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
    n_patches: int = PATCH_N,
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