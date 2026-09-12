"""
SignalScope Model Loader Module
Loads and caches the trained ResNet-50 model checkpoint.
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models

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
            "Please ensure 'best_resnet50_cifake.pth' exists in the model directory."
        )

    device = get_device()

    # Instantiate ResNet-50 without downloading pretrained weights
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)

    try:
        checkpoint = torch.load(target_path, map_location=device)
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
            # Assume dictionary itself is the state_dict
            state_dict = checkpoint
    else:
        state_dict = checkpoint

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

    except (ImportError, Exception):
        # Fallback to module-level caching for tests / scripts
        cache_key = str(resolved_path.resolve()) if resolved_path.exists() else str(resolved_path)
        if cache_key not in _LOADED_MODEL_CACHE:
            _LOADED_MODEL_CACHE[cache_key] = _load_model_impl(resolved_path)
        return _LOADED_MODEL_CACHE[cache_key]
