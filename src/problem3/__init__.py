"""Problem 3: sparse, prototype-grounded and counterfactually verified prediction."""

from .config import load_config
from .model import SEPCNet

__all__ = ["SEPCNet", "load_config"]
