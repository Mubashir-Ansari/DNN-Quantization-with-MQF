"""Unstructured pruning methods."""

from .classical.magnitude import MagnitudePruning
from .classical.random import RandomPruning
from .y2019.snip import SNIPPruning
from .y2023.wanda import WANDAPruning
from .y2023.pdp import PDPPruning

__all__ = [
    'MagnitudePruning',
    'RandomPruning', 
    'SNIPPruning',
    'WANDAPruning',
    'PDPPruning'
]