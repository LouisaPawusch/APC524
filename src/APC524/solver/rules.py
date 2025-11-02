# ---------
# IMPORTS
# ---------
from __future__ import annotations

import numpy as np

from APC524.solver.automaton import CellularAutomaton
from APC524.solver.kernels import MOORE_KERNEL

CGOL_RULES_DICT = {"dead": 0, "alive": 1}


def CGOL_init(kernel=MOORE_KERNEL, grid_size=(50, 50)):
    """
    Fully initializes a CA for Conway's Game of Life.
    Returns a CellularAutomaton object ready to step.
    """
    rng = np.random.default_rng()
    states_dict = {"dead": 0, "alive": 1}
    grid = rng.choice([states_dict["dead"], states_dict["alive"]], size=grid_size)
    nstates = len(states_dict)
    history = [grid.copy()]

    return CellularAutomaton(
        grid=grid,
        kernel=kernel,
        states_dict=states_dict,
        nstates=nstates,
        history=history,
    )


def CGOL_rules(grid=None, neighbour_counts=None, states_dict=None):
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
    states_dict : Dict[str, int]
        Dictionary defining the possible states for a cell

    Returns
    -------
    grid_update : np.ndarray
        the grid for the next step after the rules have been applied

    Example
    -------
    >>>
    """

    # apply the rules
    dead_val = states_dict["dead"]
    alive_val = states_dict["alive"]

    if neighbour_counts is None:
        counts_err = "Neighbour counts must be provided for stepping."
        raise ValueError(counts_err)

    grid_update = grid.copy()
    alive_mask = grid == alive_val
    dead_mask = grid == dead_val

    # because we only have 2 states in basic CGOL but convolve_neighbours_2D counts all states
    # we only need to take the grid counting the living cells (neighbour_counts[1])

    # cell dies of lonliness if it has less than two neighbours
    grid_update[alive_mask & (neighbour_counts[alive_val] < 2)] = dead_val
    # cell lives if it has two or more live neighbours
    grid_update[
        alive_mask
        & ((neighbour_counts[alive_val] == 2) | (neighbour_counts[alive_val] == 3))
    ] = alive_val
    # kill cell if it is overcrowded (more than three neighbours)
    grid_update[alive_mask & (neighbour_counts[alive_val] > 3)] = dead_val
    # revive a cell with exactly 3 neighbours
    grid_update[dead_mask & (neighbour_counts[alive_val] == 3)] = alive_val

    return grid_update


def disease_rules(
    grid: np.ndarray,
    neighbour_counts: np.ndarray,
    vaccine_rate: float,
    mortality_rate: float,
    vaccine_efficacy: float,
) -> np.ndarray:
    """
    Function implements rules for a disease spread simulation.

    In this simulation, the general rules for CGOL are followed,
    except that cells may now remain explicitly dead. Once they
    are dead they can no longer be infected. This rule set also
    allows for vaccines to be administered (with a certain efficacy)
    demonstrating how stochasticity can be added to a CA problem.

    Parameters
    ----------
    grid : np.ndarray
        the 2D grid on which the game is being played
    neighbour_counts: np.ndarray
        the counts for each neighbour in each state as determined
        by the convolve_neighbours_2D operation
    vaccine_rate : float
        rate at which individuals are vaccinated (between 0.0 and 1.0)
    mortality_rate : float
        mortality rate of a given disease (between 0.0 and 1.0)
    vaccine_efficacy : float
        chance that a vaccinated person will get the disease anyway (between 0.0 and 1.0)
    """
