"""
SignalScope Model Loader Module
Loads and caches the trained ResNet-50 model checkpoint.
"""


from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
import numpy

# PyTorch 2.6+ security update: allowlist numpy scalar for unpickling checkpoint
try:
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])
except Exception:
    pass

try:
    from app.config import MODEL_PATH, NUM_CLASSES
except (ImportError, ModuleNotFoundError):
    from config import MODEL_PATH, NUM_CLASSES

# Global cache for non-Streamlit environments (e.g. CLI/tests)
_LOADED_MODEL_CACHE = {}


def get_device() -> torch.device:
    """
    Automatically detect and return CUDA device if available, else CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_model_impl(target_path: Path) -> tuple[nn.Module, torch.device]:
    """
    Internal implementation to load ResNet-50 and restore checkpoint weights.
    """
    if not target_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint file not found at: {target_path}. "
            "Please ensure 'best_resnet50_cifake_retrained.pth' exists in the model directory."
        )

    device = get_device()

    try:
        checkpoint = torch.load(target_path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load checkpoint from {target_path}. File may be corrupted or unreadable. Error: {e}"
        ) from e

    # Extract state dict based on checkpoint structure
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Instantiate ResNet-50 without downloading pretrained weights
    model = models.resnet50(weights=None)

    # Detect if checkpoint uses native 32x32 CIFAR adapted stem (3x3 conv1, Identity maxpool)
    if "conv1.weight" in state_dict and state_dict["conv1.weight"].shape == torch.Size([64, 3, 3, 3]):
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)

    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(
            f"Failed to match loaded state_dict with ResNet-50 architecture. Error: {e}"
        ) from e

    model.to(device)
    model.eval()
    return model, device


def load_model(model_path: str | Path | None = None) -> tuple[nn.Module, torch.device]:
    """
    Loads the trained PyTorch model.
    Uses Streamlit caching if running inside Streamlit, or in-memory dictionary caching otherwise.
    """
    resolved_path = Path(model_path) if model_path else MODEL_PATH

    # Check if running within a Streamlit app context
    try:
        import streamlit as st

        # Use Streamlit's cache_resource for thread-safe caching across app reruns
        @st.cache_resource(show_spinner="Loading ResNet-50 SignalScope Model...")
        def _cached_streamlit_loader(path_str: str):
            return _load_model_impl(Path(path_str))

        return _cached_streamlit_loader(str(resolved_path))

    except ImportError:
        # Fallback to module-level caching for tests / scripts
        cache_key = str(resolved_path.resolve()) if resolved_path.exists() else str(resolved_path)
        if cache_key not in _LOADED_MODEL_CACHE:
            _LOADED_MODEL_CACHE[cache_key] = _load_model_impl(resolved_path)
        return _LOADED_MODEL_CACHE[cache_key]


def resolve_model_device(model: torch.nn.Module | None = None, device: torch.device | None = None) -> tuple[torch.nn.Module, torch.device]:
    """
    Helper to resolve model and device, loading them if not provided.
    Reduces boilerplate in strategy files.
    """
    if model is None or device is None:
        loaded_model, loaded_device = load_model()
        model = model or loaded_model
        device = device or loaded_device
    return model, device
