# ---------
# IMPORTS
# ---------
from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve


def convolve_neighbours_2D(
    grid: np.ndarray, kernel: np.ndarray, nstates: int
) -> np.ndarray:
    """
    Return the number of cells in a given state for a neighbourhood
    for a 2D grid.

    Parameters
    ----------
    grid : np.ndarray
        2D array of integers for a given cellular automaton grid
        representing the cell states in the user specified rules
    kernel : np.ndarray
        2D array representing the neighbours around which to search
        for a given cell
    nstates : int
        total number of states in the user specified rules

    Returns
    -------
    neighbour_counts : np.ndarray
        3D array in the shape (nstates, rows, cols) where neighbour_counts[x, i, j]
        is the number of cells in state x surrounding the cell at [i, j]

    Examples
    --------
    [add examples here]

    """
    # count the neighbours in each state
    neighbour_counts = np.zeros((nstates, *grid.shape), dtype=int)

    for state in range(nstates):
        # make a binary mask of the grid for a given cell state
        mask = (grid == state).astype(int)
        # apply convolution to count the number of cells in that state

        # use mode='constant' here so that it doesn't do wrap around behaviour
        # instead if on outside of grid, will fill with a buffer of 0s
        neighbour_counts[state] = convolve(mask, kernel, mode="constant", cval=0)

        # NOTE: at some point, we need to enforce that the states in the user rules
        # count up from 0 or pass a cval to this function to denote dead state!

    return neighbour_counts
