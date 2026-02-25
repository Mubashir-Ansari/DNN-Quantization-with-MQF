"""Classical unstructured pruning methods."""

from .magnitude import MagnitudePruning
from .random import RandomPruning

__all__ = ['MagnitudePruning', 'RandomPruning']