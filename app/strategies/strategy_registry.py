"""
SignalScope Strategy Registry Factory
Provides dynamic registration and lookup for all inference strategy modules.
"""

from typing import Dict, Type, List, Any
from PIL import Image
import torch

from app.strategies.base_strategy import BaseStrategy


class StrategyRegistry:
    """
    Singleton registry managing available prediction strategy instances.
    """
    _registry: Dict[str, BaseStrategy] = {}

    @classmethod
    def register(cls, strategy_instance: BaseStrategy) -> BaseStrategy:
        """Registers a strategy instance by its name key."""
        name = strategy_instance.name.lower()
        cls._registry[name] = strategy_instance
        return strategy_instance

    @classmethod
    def get(cls, name: str) -> BaseStrategy:
        """Retrieves a registered strategy by key name."""
        name = name.lower()
        if name not in cls._registry:
            valid_names = list(cls._registry.keys())
            raise KeyError(f"Strategy '{name}' not found in registry. Registered strategies: {valid_names}")
        return cls._registry[name]

    @classmethod
    def list_strategies(cls) -> List[Dict[str, str]]:
        """Returns list of metadata dicts for all registered strategies."""
        return [
            {
                "key": name,
                "display_name": strategy.display_name,
                "description": strategy.description,
            }
            for name, strategy in cls._registry.items()
        ]

    @classmethod
    def run_strategy(
        cls,
        name: str,
        image: Image.Image,
        model: torch.nn.Module | None = None,
        device: torch.device | None = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Convenience method to execute a named strategy."""
        strategy = cls.get(name)
        return strategy.predict(image, model=model, device=device, **kwargs)


def register_strategy(strategy_instance: BaseStrategy) -> BaseStrategy:
    """Decorator / helper function to register a strategy instance."""
    return StrategyRegistry.register(strategy_instance)


def get_strategy(name: str) -> BaseStrategy:
    """Helper function to retrieve strategy instance."""
    return StrategyRegistry.get(name)


def list_strategies() -> List[Dict[str, str]]:
    """Helper function to list all registered strategy keys and display names."""
    return StrategyRegistry.list_strategies()
