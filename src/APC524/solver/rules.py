# ---------
# IMPORTS
# ---------
from __future__ import annotations

import numpy as np

from APC524.solver.automaton import CellularAutomaton
from APC524.solver.kernels import MOORE_KERNEL

CGOL_RULES_DICT = {"dead": 0, "alive": 1}
DISEASE_RULES_DICT = {"dead": 0, "healthy": 1, "infected": 2, "immune": 3}


def CGOL_init(
    kernel=MOORE_KERNEL, grid_size=(50, 50), states_dict=CGOL_RULES_DICT, rng=None
):
    """
    Fully initializes a CA for Conway's Game of Life.
    Returns a CellularAutomaton object ready to step.

    Parameters
    ----------
    kernel : np.ndarray
        the kernel used to initialize the ca object, used by convolution
    grid_size : tuple[int, int]
        tuple of integers defining the grid size
    rng : np.random.rng
        random number generatore, default is to create inside the grid,
        usually only worth defining for testing

    Returns
    -------
        : CellularAutomaton
        CA object you created to play CGOL
    """
    rng = rng or np.random.default_rng()

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


def disease_init(
    states_dict=None,
    kernel=MOORE_KERNEL,
    grid_size=(50, 50),
    vaccine_rate=0.0,
    rng=None,
):
    """
    Function initializes a grid for the disease spread cellular
    automaton experiment

    Parameters
    -----------
    states_dict : dict[str, int]:
        dictionary defining the values associated with each cell state
    kernel : np.ndarray
        the kernel used by the rules function to update neighbours
    grid_size : tuple[int, int]
        tuple defining the size of the grid on which to play the game
    vaccine_rate : float (0.0 -> 1.0):
        the proportion of the population that are vaccinated against the
        disease at the beginning of the simulation
    rng : np.random.rng()
        numpy random number generator, usually passed in for testing only

    Returns
    -------
    ca : CellularAutomaton
        a cellular automaton object which is then used to play the game

    Examples
    --------
    >>> # Initialize a 20x20 grid with 10% vaccinated
    >>> ca = disease_init(
    ...     states_dict=DISEASE_RULES_DICT,
    ...     grid_size=(20, 20),
    ...     vaccine_rate=0.1
    ... )
    >>>
    >>> # Inspect the grid
    >>> ca.grid.shape
    (20, 20)

    """
    rng = rng or np.random.default_rng()
    states_dict = DISEASE_RULES_DICT

    # define a grid with only healthy and infected values first as in traditional CGOL
    grid = rng.choice([states_dict["infected"], states_dict["healthy"]], size=grid_size)

    # now based on the vaccine rate, make some immune
    mask = rng.random(grid.shape) < vaccine_rate
    grid[mask] = states_dict["immune"]

    return CellularAutomaton(
        grid=grid,
        kernel=kernel,
        states_dict=states_dict,
    )


def disease_rules(
    grid=None,
    neighbour_counts=None,
    states_dict=None,
    mortality_rate=None,
    vaccine_efficacy=None,
    infection_rate=None,
    recovery_rate=None,
    rebirth=False,
    rng=None,
) -> np.ndarray:
    """
    Function implements rules for a disease spread simulation.

    In this simulation, the general rules for CGOL are followed,
    except that cells may now remain explicitly dead. Once they
    are dead they can no longer be infected. This rule set also
    allows for vaccines to be administered (with a certain efficacy)
    demonstrating how stochasticity can be added to a CA problem.

    It would be interesting to add some knowledge of history to
    the problem (e.g. the longer it takes for a cell to recover
    the more likely it is to die - could do by checking history
    in the ca object for each cell and creating a multiplier for
    mortality rate here?)

    Parameters
    ----------
    grid : np.ndarray
        the 2D grid on which the game is being played
    neighbour_counts: np.ndarray
        the counts for each neighbour in each state as determined
        by the convolve_neighbours_2D operation
    states_dict : dict[Str, int]
        dictionary of states that each cell could be in as defined in the init function.
    mortality_rate : float
        mortality rate of a given disease (between 0.0 and 1.0)
    vaccine_efficacy : float
        chance that a vaccinated person will get the disease anyway (between 0.0 and 1.0)
    infection_rate : float
        chance that a person next to a vaccinated person will get infected (between 0.0 and 1.0)
    recovery_rate : float
        chance that a living, infected person will recover at the next timestep
    rebirth = bool
        should the function allow the CGOL rule of rebirth (exactly 3 living neighbours)

    Returns
    -------
    grid_update : np.ndarray
        the grid for the next step after the rules have been applied

    Examples
    --------
    >>>
    """
    # define random number generator for disease spread
    rng = rng or np.random.default_rng()

    if neighbour_counts is None:
        counts_err = "Neighbour counts must be provided for stepping."
        raise ValueError(counts_err)

    grid_update = grid.copy()

    healthy_val = states_dict["healthy"]
    infected_val = states_dict["infected"]
    immune_val = states_dict["immune"]
    dead_val = states_dict["dead"]

    # mask the cells in each state
    healthy_mask = grid == healthy_val
    infected_mask = grid == infected_val
    immune_mask = grid == immune_val
    dead_mask = grid == dead_val

    # find out how many cells have infected neighbours
    infected_neigh = neighbour_counts[infected_val]

    # SPREAD THE DISEASE!
    # 1. healthy cells can become infected if near infection and chance < infection_rate
    infection_chance_healthy = (
        healthy_mask & (infected_neigh > 0) & (rng.random(grid.shape) < infection_rate)
    )

    # 2. immune (vaccinated) cells can become infected
    infection_chance_immune = (
        immune_mask
        & (infected_neigh > 0)
        & (rng.random(grid.shape) < infection_rate * (1 - vaccine_efficacy))
    )

    # 3. some infected cells die
    death_chance = infected_mask & (rng.random(grid.shape) < mortality_rate)

    # 4. infected cells can recover (become immune)
    recovery_chance = infected_mask & (rng.random(grid.shape) < recovery_rate)

    grid_update[infection_chance_healthy | infection_chance_immune] = infected_val
    grid_update[death_chance] = dead_val
    grid_update[recovery_chance] = immune_val

    # if we are allowing birth follow CGOL birth rule
    if rebirth:
        grid_update[dead_mask & (neighbour_counts[healthy_val] == 3)] = healthy_val
    else:
        grid_update[dead_mask] = dead_val  # if no rebirth, the dead stay dead

    return grid_update
