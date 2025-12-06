"""
Example script: Display a 3x3x3 Conway's Game of Life animation.

- Each layer is shown side-by-side.
- Command to run from project root: python -m examples.display_3d_cgol
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from APC524.solver.automaton import CellularAutomaton
from APC524.solver.kernels import MOORE_KERNEL
from APC524.solver.rules import CGOL_RULES_DICT, CGOL_rules
from APC524.solver.utils import convolve_neighbours_2D

layers, rows, cols = 3, 3, 3
nstates = 2 
steps = 5
rng = np.random.default_rng(42)
initial_grid = rng.integers(0, nstates, size=(layers, rows, cols))

automata = []
for layer in range(layers):
    ca = CellularAutomaton(
        grid=initial_grid[layer],
        nstates=nstates,
        kernel=MOORE_KERNEL,
        states_dict=CGOL_RULES_DICT
    )

    ca.history = [ca.grid.copy()]
    for _ in range(steps):
        ca.step(CGOL_rules, convolve_neighbours_2D)
    automata.append(ca)

fig, axes = plt.subplots(1, layers, figsize=(layers * 3, 3))
if layers == 1:
    axes = [axes]

imgs = [ax.imshow(ca.history[0], cmap="binary", vmin=0, vmax=1) for ca, ax in zip(automata, axes)]
for i, ax in enumerate(axes):
    ax.set_title(f"Layer {i}")
    ax.axis("off")

for ax in axes:
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)

def update(frame):
    for img, ca in zip(imgs, automata):
        img.set_data(ca.history[frame])
    fig.suptitle(f"Step {frame}", fontsize=16)
    return imgs

anim = FuncAnimation(fig, update, frames=len(automata[0].history), interval=500, blit=True)
plt.show()

