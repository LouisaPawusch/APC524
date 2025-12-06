import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from APC524.solver.automaton import CellularAutomaton
from APC524.solver.kernels import MOORE_KERNEL
from APC524.solver.rules import CGOL_RULES_DICT, CGOL_rules
from APC524.solver.utils import convolve_neighbours_2D

# Dimensions
layers, rows, cols = 3, 3, 3
nstates = 2
steps = 5  # number of CA steps

# Initialize 3x3x3 grid randomly
rng = np.random.default_rng(42)
initial_grid = rng.integers(0, nstates, size=(layers, rows, cols))

# Create CellularAutomaton for each layer and record history
automata = []
for layer in range(layers):
    ca = CellularAutomaton(
        grid=initial_grid[layer],
        nstates=nstates,
        kernel=MOORE_KERNEL,
        states_dict=CGOL_RULES_DICT
    )
    ca.history = [ca.grid.copy()]  # start with initial state
    for _ in range(steps):
        ca.step(CGOL_rules, convolve_neighbours_2D)
    #    ca.history.append(ca.grid.copy())
    automata.append(ca)
    print(f"Layer {layer} history length: {len(ca.history)}")  # sanity check

# Setup figure with subplots for each layer
fig, axes = plt.subplots(1, layers, figsize=(layers * 3, 3))
if layers == 1:
    axes = [axes]

# Use matshow for visible grid lines
imgs = []
for i, (ax, ca) in enumerate(zip(axes, automata)):
    img = ax.matshow(ca.history[0], cmap="binary", vmin=0, vmax=1)
    imgs.append(img)
    ax.set_title(f"Layer {i}, Step 0")
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=1)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

# Animation update function
def update(frame):
    for i, (img, ca, ax) in enumerate(zip(imgs, automata, axes)):
        img.set_data(ca.history[frame])
        ax.set_title(f"Layer {i}, Step {frame}")
    return imgs

# Animate only over the number of steps in history
nframes = len(automata[0].history)
anim = FuncAnimation(fig, update, frames=nframes, interval=500, blit=False)
plt.show()

