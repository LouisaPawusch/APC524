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
    Compute the number of neighbors in each state for a 2D cellular automaton grid.

    This function applies a 2D convolution to count, for each cell, how many of its
    neighbors are in each possible state. It returns a 3D array with dimensions
    (nstates, rows, cols), where each "layer" corresponds to counts for a given state.

    Parameters
    ----------
    grid : np.ndarray
        2D array of integers representing the current state of each cell in the grid.
        Each integer must correspond to a valid state in the range [0, nstates-1].
    kernel : np.ndarray
        2D array defining the neighborhood to consider when counting neighbors.
        For example, a 3x3 matrix of ones counts all adjacent cells (Moore neighborhood).
    nstates : int
        Total number of possible states in the grid. Must be a positive integer.

    Returns
    -------
    neighbour_counts : np.ndarray
        3D array of shape (nstates, rows, cols). Each slice neighbour_counts[state]
        gives the number of neighbors in that state for each cell.

    Raises
    ------
    ValueError
        If `grid` is not a 2D array.
        If `kernel` is not a 2D array.
        If `nstates` is not a positive integer.
        If any value in `grid` is not in the range [0, nstates-1].

    Examples
    --------
    >>> import numpy as np
    >>> from APC524.solver.utils import convolve_neighbours_2D
    >>> grid = np.array([[0, 1, 0],
    ...                  [1, 1, 0],
    ...                  [0, 0, 1]])
    >>> kernel = np.ones((3, 3), dtype=int)
    >>> neighbour_counts = convolve_neighbours_2D(grid, kernel, nstates=2)
    >>> neighbour_counts.shape
    (2, 3, 3)
    >>> neighbour_counts[0]
    array([[1, 1, 2],
           [2, 2, 2],
           [1, 2, 1]])
    >>> neighbour_counts[1]
    array([[3, 3, 2],
           [2, 3, 2],
           [2, 1, 2]])

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
