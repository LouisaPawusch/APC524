from __future__ import annotations

from .automaton import CellularAutomaton
from .kernels import MOORE_KERNEL, MOORE_KERNEL_3D, VON_NEUMANN_KERNEL
from .rules import (
    CGOL_3D_init,
    CGOL_3D_rules,
    CGOL_init,
    CGOL_rules,
    disease_init,
    disease_rules,
)
from .utils import convolve_neighbours_2D

__all__ = [
    "CellularAutomaton",
    "CGOL_rules",
    "CGOL_init",
    "CGOL_3D_init",
    "CGOL_3D_rules",
    "convolve_neighbours_2D",
    "MOORE_KERNEL",
    "VON_NEUMANN_KERNEL",
    "MOORE_KERNEL_3D",
    "disease_rules",
    "disease_init",
]
