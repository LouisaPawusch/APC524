from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from APC524.solver import (
    CGOL_init,
    CGOL_rules,
    convolve_neighbours_2D,
)
from APC524.visualization import animate_automaton


def run_cgol_example(save_path, grid_size=(50, 50), steps=100, interval=200):
    """
    Example script to run Conway's Game of Life using the CellularAutomaton class
    and visualize it with the animate_automaton function.

    Parameters
    ----------
    save_path : str
        File path to save the animation ('.mp4' or '.gif').
    grid_size : tuple of int, optional
        Size of the simulation grid as (rows, columns). Default = (50, 50).
    steps : int
        Number of simulation steps. Default = 100.
    interval : int
        Delay between frames in milliseconds for the animation. Default = 200.
    """

    # Create CA instance
    ca = CGOL_init(grid_size=grid_size)

    # Run a few steps
    for _ in range(steps):
        ca.step(CGOL_rules, convolve_neighbours_2D)

    # Animate
    animate_automaton(ca, interval=interval, save_as=save_path)


if __name__ == "__main__":
    save_path = "cgol_animation.gif"  # Change to .mp4 if preferred
    run_cgol_example(save_path=save_path)
