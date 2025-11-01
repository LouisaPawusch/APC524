from __future__ import annotations  # MUST be first

from . import solver, visualization
from .solver.automaton import CellularAutomaton
from .solver.kernels import MOORE_KERNEL
from .solver.rules import CGOL_rules
from .solver.utils import convolve_neighbours_2D
from .visualization.visualization import animate_automaton

__all__ = [
    "solver",
    "visualization",
    "CellularAutomaton",
    "MOORE_KERNEL",
    "CGOL_rules",
    "convolve_neighbours_2D",
    "animate_automaton",
]
