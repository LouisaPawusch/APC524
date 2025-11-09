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
    Initialize a Cellular Automaton for Conway's Game of Life.

    Parameters
    ----------
    kernel : np.ndarray
        Neighborhood convolution kernel. Must be 2D.
    grid_size : tuple[int, int]
        Size of the grid. Must be positive integers.
    states_dict : dict[str, int]
        Mapping of state names to integer values. Must contain 'ALIVE' and 'DEAD'.
    rng : np.random.Generator, optional
        Random number generator. If None, `np.random.default_rng()` is used.

    Returns
    -------
    CellularAutomaton
        Initialized CA simulation.

    Examples
    --------
    >>> from APC524.solver.automaton import CGOL_init
    >>> ca = CGOL_init(grid_size=(10, 10))
    >>> ca.grid.shape
    (10, 10)
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
    Apply Conway's Game of Life rules to a grid.

    Parameters
    ----------
    grid : np.ndarray
        Current grid state.
    neighbour_counts : np.ndarray
        Count of live neighbors for each cell.
    states_dict : dict[str, int]
        Defines the cell state integer values.

    Returns
    -------
    np.ndarray
        Updated grid state.

    Raises
    ------
    ValueError
        If `grid` and `neighbour_counts` do not have matching shapes.
    KeyError
        If required states ('ALIVE', 'DEAD') are missing from `states_dict`.

    Examples
    --------
    >>> new_grid = CGOL_rules(grid, neighbour_counts, states_dict)
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
    initial_infection_rate=None,
    rng=None,
):
    """
    Initialize a cellular automaton for disease spread modeling.

    Parameters
    ----------
    states_dict : dict[str, int], optional
        Mapping of named disease states to integer values.
    kernel : np.ndarray
        Convolution kernel.
    grid_size : tuple[int, int]
        Grid dimensions.
    vaccine_rate : float
        Fraction of initially vaccinated individuals (0 to 1).
    initial_infection_rate : float, optional
        Fraction of initially infected cells. If None, defaults to 0.5.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    CellularAutomaton
        Initial disease CA system.

    Raises
    ------
    ValueError
        If `vaccine_rate` or `initial_infection_rate` are outside [0, 1].

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
    # check initial values
    if vaccine_rate > 1.0:
        err_msg = "parameter vaccine_rate must be an integer between 0 and 1"
        raise ValueError(err_msg)
    
    rng = rng or np.random.default_rng()
    states_dict = DISEASE_RULES_DICT

    # If initial_infection_rate is not provided, preserve original behavior
    if initial_infection_rate is None:
        # roughly 50% infected / 50% healthy (like before)
        grid = rng.choice(
            [states_dict["infected"], states_dict["healthy"]], size=grid_size
        )
    else:
        if initial_infection_rate > 1.0:
            err_msg = "parameter initial_infection_rate must be an integer between 0 and 1"
            raise ValueError(err_msg)
        
        # start all healthy
        grid = np.full(grid_size, states_dict["healthy"], dtype=int)
        # infect a fraction of cells
        mask_infected = rng.random(grid.shape) < initial_infection_rate
        grid[mask_infected] = states_dict["infected"]

    # now based on the vaccine rate, make some immune
    mask = rng.random(grid.shape) < vaccine_rate
    grid[mask] = states_dict["immune"]

    return CellularAutomaton(
        grid=grid,
        kernel=kernel,
        states_dict=states_dict,
        nstates=len(states_dict),
        history=[grid.copy()],
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
    Apply disease transition rules to the grid.

    Parameters
    ----------
    grid : np.ndarray
        The current state grid.
    neighbour_counts : np.ndarray
        Neighbor state counts.
    states_dict : dict[str, int]
        Disease state-to-integer mapping.
    mortality_rate : float
        Probability an infected cell dies per step.
    vaccine_efficacy : float
        Probability vaccination prevents infection.
    infection_rate : float
        Base infection prob. per infected neighbor.
    recovery_rate : float
        Probability an infected cell recovers per step.
    rebirth : bool
        Whether recovered cells can become susceptible again under CGOL rules.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    np.ndarray
        Updated grid state.

    Raises
    ------
    ValueError
        If any probability parameter is outside [0, 1].
    ValueError
        If `grid` and `neighbour_counts` shapes do not match.

    Examples
    --------
    >>> updated_grid = disease_rules(grid, neighbour_counts, states_dict,
    ...                              mortality_rate=0.01,
    ...                              vaccine_efficacy=0.8,
    ...                              infection_rate=0.2,
    ...                              recovery_rate=0.1)
    """
    for name, param in [
        ("mortality_rate", mortality_rate),
        ("vaccine_efficacy", vaccine_efficacy),
        ("infection_rate", infection_rate),
        ("recovery_rate", recovery_rate),
    ]:
        if param is None:
            continue  # allow None if not required
        if not (0 <= param <= 1):
            raise ValueError(f"{name} must be between 0 and 1, got {param}")
        
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
