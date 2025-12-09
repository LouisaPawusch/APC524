from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import BoundaryNorm, ListedColormap


def bold_axes(ax):
    """
    Sets matplotlib axes linewidths to 2, making them
    bold and more attractive

    Parameters
    -----------
    ax : mpl.Axes
        axes to be bolded
    """
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)

    # increase tick width
    ax.tick_params(width=2)

    for label in ax.get_xticklabels():
        label.set_weight("bold")

    for label in ax.get_yticklabels():
        label.set_weight("bold")


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
    ax1.set_title("Epidemic Simulation", loc="right", fontweight="bold", fontsize=14)
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
        ax2.plot([], [], color=colormap[i], label=name, lw=2)[0]
        for i, name in enumerate(states_dict)
    ]

    legend_props = {"size": 12, "weight": "bold"}
    ax2.legend(frameon=False, ncols=2, loc="upper right", prop=legend_props)

    ax2.set_xlabel("Step", fontweight="bold", fontsize=14)
    ax2.set_ylabel("Number of Cells", fontweight="bold", fontsize=14)

    # format all the lines that will be plotted and updated in the sim
    lines = [
        ax2.plot([], [], color=colormap[i], lw=2)[0]
        for i, name in enumerate(list(states_dict.keys()))
    ]

    bold_axes(ax2)
    ax2.set_title(
        "Time Evolution of Disease Spread", fontweight="bold", fontsize=14, loc="left"
    )

    def update(frame):
        img.set_data(automaton.history[frame])
        ax1.set_title(
            f"Disease Spread — Step {frame}", loc="left", fontweight="bold", fontsize=14
        )

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


def animate_disease_ensemble(ensemble, states_dict, interval=150, save_as=None):
    """
    Makes an animation of the ensemble and its statistics for
    visualization. This is slightly different from a normal CA
    simulation in that the graphs show the percent in each state.

    Parameters
    ----------
    ensemble : np.ndarray
        the array of ensemble histories
    states_dict : dict
        the dictionary giving the cell states and their integer values
    interval : float
        interval between frames for animation
    save_as : string
        path at which to save out the animation

    """
    T, M, X, Y = ensemble.shape
    state_names = list(states_dict.keys())
    nstates = len(state_names)

    counts_mean = np.zeros((T, nstates))
    counts_std = np.zeros((T, nstates))
    for t in range(T):
        for i, name in enumerate(state_names):
            member_counts = [
                (ensemble[t, m] == states_dict[name]).sum() for m in range(M)
            ]
            counts_mean[t, i] = np.mean(member_counts)
            counts_std[t, i] = np.std(member_counts)

    pct_grids = np.zeros((T, nstates, X, Y))
    for t in range(T):
        for i, name in enumerate(state_names):
            pct_grids[t, i] = np.mean(ensemble[t] == states_dict[name], axis=0) * 100

    bounds = np.arange(0, 105, 5)

    # Use magma colormap
    cmap = plt.get_cmap("magma_r")

    # Create a norm to fix the mapping of values to colormap
    norm = BoundaryNorm(bounds, ncolors=cmap.N, clip=False)

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 2])
    ax_cells = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    ax_line = fig.add_subplot(gs[2, :])

    # Top heatmaps
    im_cells = []
    for i, ax in enumerate(ax_cells):
        mean0 = pct_grids[0, i].mean()
        std0 = pct_grids[0, i].std()

        im = ax.imshow(pct_grids[0, i], cmap=cmap, norm=norm)

        ax.set_title(f"{state_names[i]}", loc="left")
        ax.set_title(f"Mean: {mean0:.2f}%   Std: {std0:.2f}%", loc="right")
        ax.axis("off")
        im_cells.append(im)
    fig.colorbar(
        im_cells[0],
        ax=ax_cells,
        orientation="vertical",
        fraction=0.02,
        pad=0.02,
        label="% members",
    )

    # Bottom line plot
    x = np.arange(T)
    colormap = ["#648FFF", "#785EF0", "#DC267F", "#FE6100"]
    lines = []
    ci_bands = []
    for i, name in enumerate(state_names):
        (line,) = ax_line.plot([], [], lw=2, color=colormap[i], label=name)
        lines.append(line)
        band = ax_line.fill_between(
            x, np.zeros(T), np.zeros(T), color=colormap[i], alpha=0.2
        )
        ci_bands.append(band)

    ax_line.set_xlim(0, T - 1)
    ax_line.set_ylim(0, (counts_mean + counts_std).max() * 1.15)
    ax_line.set_xlabel("Step")
    ax_line.set_ylabel("Number of Cells")
    ax_line.set_title("Temporal Evolution of Epidemic", loc="left")
    ax_line.legend(frameon=False, ncol=2)

    def update(frame):
        # Update top heatmaps
        for i, im in enumerate(im_cells):
            im.set_data(pct_grids[frame, i])
            meanf = pct_grids[frame, i].mean()
            stdf = pct_grids[frame, i].std()
            ax_cells[i].set_title(f"{state_names[i]}", loc="left")
            ax_cells[i].set_title(f"Mean: {meanf:.2f}%   Std: {stdf:.2f}%", loc="right")
        # Update bottom lines
        for band in ci_bands:
            band.remove()
        ci_bands.clear()
        for i, line in enumerate(lines):
            line.set_data(x[: frame + 1], counts_mean[: frame + 1, i])
            band = ax_line.fill_between(
                x[: frame + 1],
                counts_mean[: frame + 1, i] - counts_std[: frame + 1, i],
                counts_mean[: frame + 1, i] + counts_std[: frame + 1, i],
                color=colormap[i],
                alpha=0.2,
            )
            ci_bands.append(band)
        return im_cells + lines + ci_bands

    anim = FuncAnimation(fig, update, frames=T, interval=interval, blit=False)

    if save_as:
        save_as = str(save_as)
        print(f"Saving animation → {save_as}")
        if save_as.endswith(".gif"):
            anim.save(save_as, writer=PillowWriter(fps=1000 / interval))
        else:
            anim.save(save_as)
    fig.savefig(f"{save_as}.png", dpi=1000)
    return anim
