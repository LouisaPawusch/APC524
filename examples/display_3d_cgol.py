"""
This script initializes a 3D cellular automaton using Conway's Game of Life rules and visualizes its evolution over time using Matplotlib's 3D plotting capabilities.

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
from scipy.signal import convolve

from APC524.solver.kernels import MOORE_KERNEL_3D

layers, rows, cols = 3, 3, 3
nstates = 2
steps = 10

rng = np.random.default_rng()
grid = rng.integers(0, nstates, size=(layers, rows, cols))

kernel = MOORE_KERNEL_3D
history = [grid.copy()]


def step(grid):
    neighbor_count = convolve(grid, kernel, mode="same")
    new_grid = np.zeros_like(grid)
    new_grid[
        (grid == 1)
        & ((neighbor_count == 4) | (neighbor_count == 5) | (neighbor_count == 6))
    ] = 1
    new_grid[(grid == 0) & ((neighbor_count == 5) | (neighbor_count == 6))] = 1
    return new_grid


for _ in range(steps):
    grid = step(grid)
    history.append(grid.copy())

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
plt.show()
