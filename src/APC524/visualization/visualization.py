from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def animate_automaton(automaton, interval: int = 200, save_as: str | None = None):
    """
    Animates the evolution of a cellular automaton.

    Parameters
    ----------
    automaton : object
        An instance of a cellular automaton class with a `step` method and a `grid` attribute.
    interval : int, optional
        Time in milliseconds between frames, by default 200.
    save_as : str or None, optional
        If provided, saves the animation to the specified file, by default None.

    Returns
    -------
    FuncAnimation
        The animation object.
    """
    assert automaton is not None, "Automaton instance must be provided."

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Conway's Game of Life")
    img = ax.imshow(automaton.history[0], cmap="binary")
    ax.axis("off")

    def update(frame):
        img.set_data(automaton.history[frame])
        ax.set_title(f"Conway's Game of Life — Step {frame}")
        return [img]

    anim = FuncAnimation(
        fig, update, frames=len(automaton.history), interval=interval, repeat=True
    )

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


def animate_disease(
    automaton, states_dict, interval: int = 200, save_as: str | None = None
):
    """
    Elaborates on the basic automaton animation to show the disease spread
    example

    NOTE: these animation functions are a good example of why it might
    be good to make objects for each example that inherit from each other?
    That way we can basically abstract over having 1000 functions to
    animate each different experiment that we run?

    Parameters
    ----------
    automaton : object
        An instance of a cellular automaton class with a `step` method and a `grid` attribute.
    states_dict : dict[str, int]
        dictionary containing the keys for each numerical state (used for colormap and line graph)
    interval : int, optional
        Time in milliseconds between frames, by default 200.
    save_as : str or None, optional
        If provided, saves the animation to the specified file, by default None.

    Returns
    -------
    FuncAnimation
        The animation object.
    """
    if automaton is None:
        err_msg = "Automaton instance must be provided"
        raise ValueError(err_msg)

    # colorblind friendly map (Tol light)
    colormap = ["#DDDDDD", "#44BB99", "#EE8866", "#99DDFF"]
    cmap = ListedColormap(colormap)

    # plotting the heatmap
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    ax1.set_title("Epidemic Simulation")
    img = ax1.imshow(automaton.history[0], cmap=cmap, vmin=0, vmax=len(states_dict) - 1)
    ax1.axis("off")

    # Line plot subplot
    nsteps = len(automaton.history)
    nstates = len(states_dict)

    # Precompute counts over time
    counts_over_time = np.zeros((nsteps, nstates), dtype=int)
    for t, grid in enumerate(automaton.history):
        unique, counts = np.unique(grid, return_counts=True)
        for u, c in zip(unique, counts, strict=False):
            counts_over_time[t, u] = c

    ax2.set_xlim(0, nsteps - 1)  # fixed x-axis
    ax2.set_ylim(0, counts_over_time.max() * 1.1)  # leave some vertical margin
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Number of Cells")

    lines = [
        ax2.plot([], [], color=colormap[i], label=name)[0]
        for i, name in enumerate(states_dict)
    ]

    ax2.set_xlabel("Step")
    ax2.set_ylabel("Number of Cells")
    lines = [
        ax2.plot([], [], color=colormap[i], label=name, lw=2)[0]
        for i, name in enumerate(list(states_dict.keys()))
    ]

    # spoof legend
    legend_elements = [
        Patch(color=colormap[i], label=name) for i, name in enumerate(states_dict)
    ]
    ax1.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")

    def update(frame):
        img.set_data(automaton.history[frame])
        ax1.set_title(f"Disease Spread — Step {frame}")

        # update lines
        xdata = np.arange(frame + 1)
        for i, line in enumerate(lines):
            ydata = counts_over_time[: frame + 1, i]
            line.set_data(xdata, ydata)
        return [img, *lines]

    fig.tight_layout()

    anim = FuncAnimation(
        fig,
        update,
        frames=len(automaton.history),
        interval=interval,
        blit=True,
        repeat=True,
    )

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
