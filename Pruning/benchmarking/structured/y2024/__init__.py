"""2024 structured pruning methods."""

from .torque import TorqueBasedPruning
from .adasap import AdaSAPPruning

__all__ = ['TorqueBasedPruning', 'AdaSAPPruning']