from __future__ import annotations

from .automaton import CellularAutomaton
from .kernels import MOORE_KERNEL, VON_NEUMANN_KERNEL
from .rules import CGOL_init, CGOL_rules
from .utils import convolve_neighbours_2D

__all__ = [
    "CellularAutomaton",
    "CGOL_rules",
    "CGOL_init",
    "convolve_neighbours_2D",
    "MOORE_KERNEL",
    "VON_NEUMANN_KERNEL",
]
