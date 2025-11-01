from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent / "src"))

from APC524.solver import (
    MOORE_KERNEL,
    CellularAutomaton,
    CGOL_rules,
    convolve_neighbours_2D,
)
from APC524.visualization import animate_automaton


def run_cgol_example(save_path, grid_size=(50, 50), steps=100, interval=200):
    """
    Example script to run Conway's Game of Life using the CellularAutomaton class
    and visualize it with the animate_automaton function.
    """
    # Initialize random grid (switch to numpy random number generator for Ruff compliance)
    rng = np.random.default_rng()
    grid = rng.choice([0, 1], size=grid_size)

    # Create CA instance
    ca = CellularAutomaton(grid=grid, nstates=2, kernel=MOORE_KERNEL)

    # Run a few steps
    for _ in range(steps):
        ca.step(CGOL_rules, convolve_neighbours_2D)

    # Animate
    animate_automaton(ca, interval=interval, save_as=save_path)


if __name__ == "__main__":
    save_path = "cgol_animation.gif"  # Change to .mp4 if preferred
    run_cgol_example(save_path=save_path)
