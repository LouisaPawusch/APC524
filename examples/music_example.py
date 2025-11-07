from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))
from APC524.solver import (
    CGOL_init,
    CGOL_rules,
    convolve_neighbours_2D,
)

from APC524.visualization import animate_automaton
from APC524.sonification.sonification import sonify_automaton, merge_audio_video

def run_sonified_cgol_example(
    save_name: str,
    grid_size=(50, 50),
    steps=100,
    interval=200):
    """
    Example script to run Conway's Game of Life using the CellularAutomaton class,
    visualize it, and sonify its evolution.

    Parameters
    ----------
    save_path : str
        Path where the animation will be saved (.mp4 or .gif).
    grid_size : tuple[int, int]
        Grid dimensions.
    steps : int
        Number of evolution steps.
    interval : int
        Milliseconds between frames.
    play_audio : bool
        Whether to play the generated tones.
    """

    print("Initializing Conway's Game of Life...")

    ca = CGOL_init(grid_size=grid_size)

    for _ in range(steps):
        ca.step(CGOL_rules, convolve_neighbours_2D)

    anim_path = str(save_name + ".gif")
    audio_path = str(save_name + ".wav")
    final_path = str(save_name + "_with_audio.mp4")

    print("Creating animation and audio...")

    animate_automaton(ca, interval=interval, save_as=anim_path)
    sonify_automaton(ca, interval=interval, save_audio_as=str(audio_path))

    merge_audio_video(str(anim_path), str(audio_path), str(final_path))

    print(f"✅ Sonified Game of Life saved as {save_name}")


if __name__ == "__main__":
    save_name = "cgol_sonification"
    run_sonified_cgol_example(save_name=save_name)