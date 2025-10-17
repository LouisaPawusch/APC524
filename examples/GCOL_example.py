import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from APC524.solver import CellularAutomaton, CGOL_rules, convolve_neighbours_2D, MOORE_KERNEL
from APC524.visualization import animate_automaton

def run_cgol_example(save_path, grid_size=(50, 50), steps=100, interval=200):
    """
    Example script to run Conway's Game of Life using the CellularAutomaton class
    and visualize it with the animate_automaton function.
    """
    # Initialize random grid
    grid = np.random.choice([0, 1], size=grid_size)

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