from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def animate_automaton_3D(automaton, interval: int = 500, save_as: str | None = None):
    """
    Animate a 3D cellular automaton (3x3x3 CGOL).

    Parameters
    ----------
    automaton : object
        CellularAutomaton instance with a `.history` attribute containing 3D grids.
    interval : int
        Time in milliseconds between frames, by default 500.
    save_as : str or None
        File name to save animation, supports .mp4 or .gif, by default None.

    Returns
    -------
    FuncAnimation
        Matplotlib animation object
    """
    assert automaton is not None, "Automaton instance must be provided."

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    Z, Y, X = automaton.history[0].shape
    x, y, z = np.indices((X, Y, Z))

    ax.scatter([], [], [], c=[], cmap="binary", s=100)
    ax.set_xlim(0, X - 1)
    ax.set_ylim(0, Y - 1)
    ax.set_zlim(0, Z - 1)
    ax.set_title("3D Game of Life")

    def update(frame):
        ax.cla()
        grid = automaton.history[frame]
        # get coordinates of live cells (1 = alive)
        alive = np.argwhere(grid == 1)
        if alive.size > 0:
            ax.scatter(
                alive[:, 2],
                alive[:, 1],
                alive[:, 0],  # X, Y, Z
                c="black",
                s=100,
            )
        ax.set_xlim(0, X - 1)
        ax.set_ylim(0, Y - 1)
        ax.set_zlim(0, Z - 1)
        ax.set_title(f"3D Game of Life — Step {frame}")
        return (ax,)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(automaton.history),
        interval=interval,
        blit=False,
        repeat=True,
    )
    save_as = "game_of_life_3d.gif"

    if save_as:
        print(f"Saving animation to {save_as}...")
        if save_as.endswith(".mp4"):
            anim.save(save_as, writer="ffmpeg")
        elif save_as.endswith(".gif"):
            anim.save(save_as, writer="imagemagick")
        else:
            err_msg = "File format not supported. Use .mp4 or .gif"
            raise ValueError(err_msg)

    plt.show()
