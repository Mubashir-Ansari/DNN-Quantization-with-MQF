"""2023 unstructured pruning methods."""

from .wanda import WANDAPruning
from .pdp import PDPPruning

__all__ = ['WANDAPruning', 'PDPPruning']