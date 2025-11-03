from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from APC524.solver import (
    convolve_neighbours_2D,
    disease_init,
    disease_rules,
)
from APC524.visualization import animate_disease


def run_disease_example(save_path, grid_size=(50, 100), steps=100, interval=200):
    """
    Example script to run Conway's Game of Life using the CellularAutomaton class
    and visualize it with the animate_automaton function.
    """
    # define parameters for simulation
    mortality_rate = 0.33
    vaccine_efficacy = 0.8
    infection_rate = 0.25
    recovery_rate = 0.1
    initial_infection_rate = 0.05
    initial_vax_rate = 0.1

    # Create CA instance
    ca = disease_init(
        grid_size=grid_size,
        vaccine_rate=initial_vax_rate,
        initial_infection_rate=initial_infection_rate,
    )

    # Run a few steps
    for _ in range(steps):
        ca.step(
            disease_rules,
            convolve_neighbours_2D,
            mortality_rate=mortality_rate,
            vaccine_efficacy=vaccine_efficacy,
            infection_rate=infection_rate,
            recovery_rate=recovery_rate,
        )

    # Animate
    animate_disease(ca, ca.states_dict, interval=interval, save_as=save_path)


if __name__ == "__main__":
    save_path = "examples/disease_animation.gif"  # Change to .mp4 if preferred
    run_disease_example(save_path=save_path)
