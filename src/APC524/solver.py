"""
solver.py

This file contains the basic solver for Conway's Game of Life and will be developed 
to add additional levels of complexity to the solver as needed. 

The code for this solver was inspired by the cellular automaton forest fires project at 
https://github.com/larantt/clasp410tobiastarsh/tree/main/labs/lab01 and the Game of Life
with 2D Convolution at https://gist.github.com/mikelane/89c580b7764f04cf73b32bf4e94fd3a3

The basic rules of Conway's Game of Life for any type of grid are as follows:
* If a live cell with fewer than two live neighbours exists, it dies of lonliness :(
* If a live cell has two or three live neighbours, it lives :)
* If a live cell has more than three live neighbours, it dies of starvation :(
* If a cell is currently dead and it has exactly three neighbours, a new living cell is born

(add more detail with further development)

TO DO:
- [ ] Modularise this code more (maybe split the helpers into new file etc)
- [ ] Further development of the grid function to allow irregular/multidimensional grids
- [ ] Add more complete, cleaner way of specifying rules aside from the default rules function

"""
# ---------
# IMPORTS 
# ---------
import dataclasses
from typing import Callable
import numpy as np
from scipy.ndimage import convolve

# -------------------------------------------
# Global kernels (REFACTOR TO UTILS LATER??)
# --------------------------------------------
MOORE_KERNEL = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
], dtype=int)

VON_NEUMANN_KERNEL = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=int)

# --------------------------------------------
# function to count neighbors via convolution 
# --------------------------------------------
def convolve_neighbours_2D(grid: np.ndarray, kernel : np.ndarray, nstates: int) -> np.ndarray:
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
    neighbour_counts = np.zeros((nstates,) + grid.shape, dtype=int)

    for state in range(nstates):
        # make a binary mask of the grid for a given cell state
        mask = (grid == state).astype(int)
        # apply convolution to count the number of cells in that state

        # use mode='constant' here so that it doesnt do wrap around behaviour
        # instead if on outside of grid, will fill with a buffer of 0s
        neighbour_counts[state] = convolve(mask, kernel, mode='constant', cval=0)

        # NOTE: at some point, we need to enforce that the states in the user rules
        # count up from 0 or pass a cval to this function to denote dead state!

    return neighbour_counts
