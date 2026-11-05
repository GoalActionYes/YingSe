"""Shared numerical packets. Arrays use row-major order."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class FeaturePacket:
    mode: int
    values: np.ndarray
    diagnostics: dict

@dataclass(frozen=True)
class Mapping:
    """Two RGB-axis layers, shape (2, 64, 64, 64, 3)."""
    layers: np.ndarray
    diagnostics: dict
