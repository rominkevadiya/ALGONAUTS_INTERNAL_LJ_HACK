"""
SignalScope Base Strategy Abstract Interface
Defines standard interface and output schema validation for all inference strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from PIL import Image
import torch


class BaseStrategy(ABC):
    """
    Abstract Base Class for SignalScope inference strategies.
    All strategy implementations should inherit from this class and implement the `predict` method.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique key name for strategy registry lookup (e.g., 'multiscale')."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """User-friendly display title for UI elements (e.g., 'Multi-Scale 3-Branch Analysis')."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Brief description of how the strategy operates."""
        pass

    @abstractmethod
    def predict(
        self,
        image: Image.Image,
        model: torch.nn.Module | None = None,
        device: torch.device | None = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Executes prediction strategy on the given image.

        Returns standard dictionary with required keys:
            - label: 'REAL' | 'FAKE' | 'UNCERTAIN'
            - confidence: float (0.0 to 1.0)
            - fake_probability: float (0.0 to 1.0)
            - real_probability: float (0.0 to 1.0)
            - inference_mode: str
        """
        pass


REQUIRED_OUTPUT_KEYS = {
    "label",
    "confidence",
    "fake_probability",
    "real_probability",
    "inference_mode",
    "threshold",
}


def validate_strategy_output(res: Dict[str, Any], strategy_name: str = "Strategy") -> Dict[str, Any]:
    """
    Validates that a strategy output dictionary contains all required fields and compliant types.
    """
    missing = REQUIRED_OUTPUT_KEYS - set(res.keys())
    if missing:
        raise ValueError(f"Strategy output from '{strategy_name}' is missing required keys: {missing}")

    # Ensure label compliance
    if res["label"] not in ("REAL", "FAKE", "UNCERTAIN"):
        raise ValueError(f"Invalid label '{res['label']}' from '{strategy_name}'. Must be REAL, FAKE, or UNCERTAIN.")

    # Clamp probabilities [0.0, 1.0]
    res["fake_probability"] = max(0.0, min(1.0, float(res["fake_probability"])))
    res["real_probability"] = max(0.0, min(1.0, float(res["real_probability"])))
    res["confidence"] = max(0.0, min(1.0, float(res["confidence"])))

    return res
