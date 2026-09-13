"""
SignalScope Strategies Package
Exposes BaseStrategy, StrategyRegistry, and concrete prediction strategies.
"""

from app.strategies.base_strategy import BaseStrategy, validate_strategy_output
from app.strategies.strategy_registry import (
    StrategyRegistry,
    register_strategy,
    get_strategy,
    list_strategies,
)

# Import concrete strategies to trigger self-registration
from app.strategies.resize.resize_strategy import ResizeStrategy, predict_image
from app.strategies.patch.patch_strategy import PatchStrategy, predict_image_patch_vote
from app.strategies.hybrid.hybrid_strategy import HybridStrategy, predict_image_hybrid
from app.strategies.multiscale.multiscale_strategy import MultiScaleStrategy, predict_image_multiscale
from app.strategies.tta.tta_strategy import TTAStrategy, predict_image_tta
from app.strategies.auto.auto_strategy import AutoStrategy, predict_image_auto, predict_batch

__all__ = [
    "BaseStrategy",
    "validate_strategy_output",
    "StrategyRegistry",
    "register_strategy",
    "get_strategy",
    "list_strategies",
    "ResizeStrategy",
    "PatchStrategy",
    "HybridStrategy",
    "MultiScaleStrategy",
    "TTAStrategy",
    "AutoStrategy",
    "predict_image",
    "predict_image_patch_vote",
    "predict_image_hybrid",
    "predict_image_multiscale",
    "predict_image_tta",
    "predict_image_auto",
    "predict_batch",
]
