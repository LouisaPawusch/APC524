# ---------
# IMPORTS
# ---------
from __future__ import annotations

import numpy as np


def CGOL_rules(grid: np.ndarray, neighbour_counts: np.ndarray) -> np.ndarray:
    """
    Function lays out the rules for basic CGOL and determines
    what happens to each cell in the grid.

    NOTE: I would like to generalize this more and move the
    generation of the new grid to the overall class. This would
    allow the user to define their own rule set without having to
    do all the work of deciding how to assign the grid...

    Parameters
    ----------
    grid : np.ndarray
        the 2D grid on which the game is being played
    neighbour_counts : np.ndarray
        the counts for each neighbour in each state as determined
        by convolve_neighbours_2D

    Returns
    -------
    grid_update : np.ndarray
        the grid for the next step after the rules have been applied

    Example
    -------
    >>>
    """
    # make a copy of the grid to be updated in accordance with rules
    grid_update = np.copy(grid)

    # make boolean masks of the grid states
    alive = grid == 1
    dead = grid == 0

    # because we only have 2 states in basic CGOL but convolve_neighbours_2D counts all states
    # we only need to take the grid counting the living cells (neighbour_counts[1])

    # cell dies of lonliness if it has less than two neighbours
    grid_update[alive & (neighbour_counts[1] < 2)] = 0
    # cell lives if it has two or more live neighbours
    grid_update[alive & ((neighbour_counts[1] == 2) | (neighbour_counts[1] == 3))] = 1
    # kill cell if it is overcrowded (more than three neighbours)
    grid_update[alive & (neighbour_counts[1] > 3)] = 0
    # revive a cell with exactly 3 neighbours
    grid_update[dead & (neighbour_counts[1] == 3)] = 1

    return grid_update
