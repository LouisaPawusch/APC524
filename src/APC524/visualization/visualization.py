from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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
