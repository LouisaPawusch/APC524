"""
This script initializes a 3D cellular automaton using Conway's Game of Life rules and
 visualizes its evolution over time.

Notes
-----
- The simulation uses a 3D Moore neighborhood (26 neighbors) to determine cell survival and birth.
- Current rules:
    * Survival: a living cell survives if it has 4, 5, or 6 neighbors.
    * Birth: a dead cell becomes alive if it has 5 or 6 neighbors.

Example
-------
 >>> python -m examples.display_3d_cgol
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from APC524.solver import (
    CGOL_3D_init,
    CGOL_3D_rules,
    convolve_neighbours_2D,
)


def run_CGOL_3D_example(grid_size=(3, 3, 3), steps=10, save_as=None):
    """
    Example script to run Conway's Game of Life in 3D using the CellularAutomaton class
    and visualize it with the animate_automaton function.

    Parameters
    ----------
    grid_size : tuple of int, optional
        Size of the simulation grid as (rows, columns). Default = (50, 100).
    steps : int
        Number of simulation steps to run. Default = 10.
    save_as : string
        path to save the gif output
    """
    ca = CGOL_3D_init(grid_size=grid_size, rng_seed=123)

    history = ca.history
    layers, rows, cols = history[0].shape

    for _ in range(steps):
        ca.step(CGOL_3D_rules, convolve_neighbours_2D)

    fig, axes = plt.subplots(1, layers, figsize=(layers * 3, 3))
    if layers == 1:
        axes = [axes]

    imgs = []
    for i, ax in enumerate(axes):
        img = ax.matshow(history[0][i], cmap="binary", vmin=0, vmax=1)
        ax.set_title(f"Layer {i}, Step 0")
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=1)
        ax.tick_params(
            which="both", bottom=False, left=False, labelbottom=False, labelleft=False
        )
        imgs.append(img)

    def update(frame):
        for i, img in enumerate(imgs):
            img.set_data(history[frame][i])
            axes[i].set_title(f"Layer {i}, Step {frame}")
        return imgs

    anim = FuncAnimation(fig, update, frames=len(history), interval=500, blit=False)

    if save_as:
        print(f"Saving animation to {save_as}...")
        if save_as.endswith(".mp4"):
            anim.save(save_as, writer="ffmpeg")
        elif save_as.endswith(".gif"):
            anim.save(save_as, writer="imagemagick")
        else:
            err_msg = "File format not supported. Use .mp4 or .gif"
            raise ValueError(err_msg)

    return anim


if __name__ == "__main__":
    save_path = "/home/lt0663/Documents/classes/APC524/projectfigs/3d_cgol.gif"
    run_CGOL_3D_example(save_as=save_path)
