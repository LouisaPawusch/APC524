from __future__ import annotations

import numpy as np
import pytest

from APC524.solver.kernels import MOORE_KERNEL
from APC524.solver.rules import (
    CGOL_RULES_DICT,
    DISEASE_RULES_DICT,
    CGOL_init,
    CGOL_rules,
    disease_init,
    disease_rules,
)

# ------------------------------
# Test initialization functions
# ------------------------------

### CGOL FUNCTIONS ###
EXPECTED_GRIDS_CGOL_3x3 = {
    42: np.array([[0, 1, 1], [0, 0, 1], [0, 1, 0]]),
    32: np.array([[1, 0, 1], [1, 0, 0], [0, 0, 1]]),
}

EXPECTED_GRIDS_CGOL_2x5 = {
    42: np.array([[0, 1, 1, 0, 0], [1, 0, 1, 0, 0]]),
    32: np.array([[1, 0, 1, 1, 0], [0, 0, 0, 1, 1]]),
}


@pytest.fixture(params=[42, 32])
def random_seed(request):
    """
    returns a random seed to check for different
    CGOL initializations (checks stochastic behavior)
    """
    return request.param


def test_CGOL_init_3x3(random_seed):
    """
    tests the initialization function for CGOL_init using
    a random seed to ensure the grid is initialized correctly
    for a 3x3 grid.

    Parameters
    ----------
    random_seed : pytest.fixture[float]
        different random seeds to use to ensure the random
        initialization starts correctly
    """

    rng = np.random.default_rng(random_seed)

    ca = CGOL_init(
        states_dict=CGOL_RULES_DICT,
        kernel=MOORE_KERNEL,
        grid_size=(3, 3),
        rng=rng,
    )

    expected_grid = EXPECTED_GRIDS_CGOL_3x3[random_seed]
    np.testing.assert_array_equal(ca.grid, expected_grid)


def test_CGOL_init_2x5(random_seed):
    """
    tests the initialization function for CGOL_init using
    a random seed to ensure the grid is initialized correctly
    for a 2x5 grid.

    Parameters
    ----------
    random_seed : pytest.fixture[float]
        different random seeds to use to ensure the random
        initialization starts correctly
    """

    rng = np.random.default_rng(random_seed)

    ca = CGOL_init(
        states_dict=CGOL_RULES_DICT,
        kernel=MOORE_KERNEL,
        grid_size=(2, 5),
        rng=rng,
    )

    expected_grid = EXPECTED_GRIDS_CGOL_2x5[random_seed]
    np.testing.assert_array_equal(ca.grid, expected_grid)


### DISEASE SPREAD FUNCTIONS ##
EXPECTED_GRIDS_DISEASE = {
    0.0: np.array([[2, 1, 1], [2, 2, 1], [2, 1, 2]]),
    0.33: np.array([[2, 1, 1], [3, 2, 1], [2, 1, 2]]),
    0.5: np.array([[2, 1, 1], [3, 3, 3], [2, 1, 2]]),
    0.66: np.array([[2, 1, 1], [3, 3, 3], [2, 3, 2]]),
    1.0: np.array([[3, 3, 3], [3, 3, 3], [3, 3, 3]]),
}


@pytest.fixture(params=[0.0, 0.33, 0.5, 0.66, 1.0])
def vaccine_rate(request):
    """
    returns a vaccine rate for the test_disease_init
    function to use (checks stochastic behavior)
    """
    return request.param


def test_disease_init(vaccine_rate):
    """
    tests the initialization function for test_disease_init using
    a random seed to ensure the grid is initialized correctly.

    Parameters
    ----------
    vaccine_rate : pytest.fixture[float]
        different vaccination rates to use to ensure the random
        initialization starts correctly
    """

    seed = 42
    rng = np.random.default_rng(seed)

    ca = disease_init(
        states_dict=DISEASE_RULES_DICT,
        kernel=MOORE_KERNEL,
        grid_size=(3, 3),
        vaccine_rate=vaccine_rate,
        rng=rng,
    )

    expected_grid = EXPECTED_GRIDS_DISEASE[vaccine_rate]
    np.testing.assert_array_equal(ca.grid, expected_grid)


# ----------------------------
# Test CGOL Rules
# ----------------------------
@pytest.fixture
def sample_grid_2_states():
    """
    Creates a sample 3 x 3 grid for testing the rules function on
    a sample grid with two states for basic CGOL. In this grid, dead is
    0 and alive is 1.

    """
    return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int), CGOL_RULES_DICT


## check all neighbour rules for CGOL ##
def test_CGOL_rules_underpopulation(sample_grid_2_states):
    """
    Tests rule 1: Cells with less than 2 neighbours die
    of lonliness :(

    Parameters
    ----------
    sample_grid_2_states : np.ndarray
        the original sample grid

    """
    sample_grid, rules_dict = sample_grid_2_states
    grid = sample_grid.copy()
    # make an array with the same shape as the output from the 2D convolution
    counts = np.zeros((2, 3, 3), dtype=int)

    counts[1, 1, 1] = 1  # give the center one living neighbour
    result = CGOL_rules(grid, counts, rules_dict)

    assert result[1, 1] == 0  # be sure center dies of loneliness


def test_CGOL_rules_survival(sample_grid_2_states):
    """
    Tests rule 2: cells with two or more neighbours survive

    Parameters
    ----------
    sample_grid_2_states : np.ndarray
        the original sample grid

    """
    sample_grid, rules_dict = sample_grid_2_states
    grid = sample_grid.copy()
    counts = np.zeros((2, 3, 3), dtype=int)

    counts[1, 1, 1] = 2
    counts[1, 1, 2] = 3
    result = CGOL_rules(grid, counts, rules_dict)

    assert result[1, 1] == 1
    assert result[1, 2] == 1


def test_CGOL_rules_overcrowding(sample_grid_2_states):
    """
    Tests rule 3: cells with more than 3 live neighbours die from
    overcrowding

    Parameters
    ----------
    sample_grid_2_states : np.ndarray
        the original sample grid
    """
    sample_grid, rules_dict = sample_grid_2_states
    grid = sample_grid.copy()
    counts = np.zeros((2, 3, 3), dtype=int)

    counts[1, 1, 1] = 4
    result = CGOL_rules(grid, counts, rules_dict)

    assert result[1, 1] == 0


def test_CGOL_rules_reproduction(sample_grid_2_states):
    """
    Tests rule 4: DEAD cells with exactly 3 live neighbours are reborn

    Parameters
    ----------
    sample_grid_2_states : np.ndarray
        the original sample grid
    """
    sample_grid, rules_dict = sample_grid_2_states
    grid = sample_grid.copy()
    grid[1, 1] = 0  # ensure dead

    counts = np.zeros((2, 3, 3), dtype=int)
    counts[1, 1, 1] = 3

    result = CGOL_rules(grid, counts, rules_dict)
    assert result[1, 1] == 1


## check all neighbour rules for disease spread ##
@pytest.fixture
def sample_grid_disease():
    """
    Creates a sample 3x3 grid for testing the rules function on
    a sample grid with four states for disease spread.
    """
    return np.array([[1, 1, 1], [3, 2, 1], [1, 1, 1]]), DISEASE_RULES_DICT


def test_disease_rules_spread(sample_grid_disease):
    """
    Tests rule 1: healthy cells become infected if they are near the
    infection and the random chance for that cell is greater than the
    infection rate

    Parameters
    ----------
    sample_grid_disease : pytest.fixture
        fixture that generates the sample grid
    """
    sample_grid, rules_dict = sample_grid_disease

    # make an array with the same shape as the output from the 2D convolution
    counts = np.zeros((4, 3, 3), dtype=int)  # neighbour counts for each 4 states
    grid = sample_grid.copy()
    # define cells which neighbour the infected cell (von neumann example)
    counts[2] = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    result = disease_rules(
        grid,
        counts,
        rules_dict,
        mortality_rate=1.0,
        vaccine_efficacy=1.0,
        infection_rate=1.0,
        recovery_rate=0.0,
    )

    # define the expected outcome of one step
    expected = np.array([[1, 2, 1], [3, 0, 2], [1, 2, 1]])

    np.testing.assert_array_equal(result, expected)


def test_disease_rules_immunity(sample_grid_disease):
    """
    Tests rule 2: immune cells can become infected if
    vaccine efficacy is low (here 0.0)

    Parameters
    ----------
    sample_grid_disease : pytest.fixture
        fixture that generates the sample grid
    """
    sample_grid, rules_dict = sample_grid_disease

    # make an array with the same shape as the output from the 2D convolution
    counts = np.zeros((4, 3, 3), dtype=int)
    grid = sample_grid.copy()
    # define cells which neighbour the infected cell (von neumann example)
    counts[2] = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    result = disease_rules(
        grid,
        counts,
        rules_dict,
        mortality_rate=1.0,
        vaccine_efficacy=0.0,
        infection_rate=1.0,
        recovery_rate=0.0,
    )

    # define the expected outcome of one step
    expected = np.array([[1, 2, 1], [2, 0, 2], [1, 2, 1]])

    np.testing.assert_array_equal(result, expected)


def test_disease_rules_efficacy(sample_grid_disease):
    """
    Tests rule 3: immune cells can become infected if
    vaccine efficacy is low (here 0.0)

    Parameters
    ----------
    sample_grid_disease : pytest.fixture
        fixture that generates the sample grid
    """
    sample_grid, rules_dict = sample_grid_disease

    # make an array with the same shape as the output from the 2D convolution
    counts = np.zeros((4, 3, 3), dtype=int)
    grid = sample_grid.copy()
    # define cells which neighbour the infected cell (von neumann example)
    counts[2] = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    result = disease_rules(
        grid,
        counts,
        rules_dict,
        mortality_rate=1.0,
        vaccine_efficacy=0.0,
        infection_rate=1.0,
        recovery_rate=0.0,
    )

    # define the expected outcome of one step
    expected = np.array([[1, 2, 1], [2, 0, 2], [1, 2, 1]])

    np.testing.assert_array_equal(result, expected)


def test_disease_rules_stochastic_infection(sample_grid_disease):
    """
    Test that infection spreads probabilistically.

    Parameters
    ----------
    sample_grid_disease : pytest.fixture
        fixture that generates the sample grid
    """

    grid, states_dict = sample_grid_disease
    nstates = len(states_dict)
    counts = np.zeros((nstates, 3, 3), dtype=int)
    counts[states_dict["infected"]] = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    rng = np.random.default_rng(123)
    infection_rate = 0.5

    results = []
    # run 100 times to check probabilistic behavior
    for _ in range(100):
        result = disease_rules(
            grid.copy(),
            neighbour_counts=counts,
            states_dict=states_dict,
            mortality_rate=0.0,
            recovery_rate=0.0,
            infection_rate=infection_rate,
            vaccine_efficacy=1.0,
            rng=rng,
        )
        results.append(result)

    results = np.stack(results)

    # Check that some but not all neighbors got infected
    infected_counts = np.sum(results == states_dict["infected"], axis=(1, 2))
    assert np.any(infected_counts > 0)
    assert np.any(infected_counts < np.prod(grid.shape))


def test_disease_rules_stochastic_death(sample_grid_disease):
    """
    Test that infected cells die probabilistically.

    Parameters
    ----------
    sample_grid_disease : pytest.fixture
        fixture that generates the sample grid
    """
    grid, states_dict = sample_grid_disease
    nstates = len(states_dict)
    counts = np.zeros((nstates, 3, 3), dtype=int)
    counts[states_dict["infected"]] = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    rng = np.random.default_rng(123)
    mortality_rate = 0.3
    results = []

    for _ in range(100):
        result = disease_rules(
            grid.copy(),
            neighbour_counts=counts,
            states_dict=states_dict,
            mortality_rate=mortality_rate,
            recovery_rate=0.0,
            infection_rate=0.0,
            vaccine_efficacy=1.0,
            rng=rng,
        )
        results.append(result)

    results = np.stack(results)
    dead_counts = np.sum(results == states_dict["dead"], axis=(1, 2))
    assert np.any(dead_counts > 0)
    assert np.any(dead_counts < np.prod(grid.shape))


def test_disease_rules_stochastic_recovery(sample_grid_disease):
    """
    Test that infected cells can recover probabilistically.
    Parameters
    ----------
    sample_grid_disease : pytest.fixture
        fixture that generates the sample grid
    """
    grid, states_dict = sample_grid_disease
    nstates = len(states_dict)
    counts = np.zeros((nstates, 3, 3), dtype=int)
    counts[states_dict["infected"]] = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    rng = np.random.default_rng(123)
    recovery_rate = 0.5
    results = []

    for _ in range(100):
        result = disease_rules(
            grid.copy(),
            neighbour_counts=counts,
            states_dict=states_dict,
            mortality_rate=0.0,
            recovery_rate=recovery_rate,
            infection_rate=0.0,
            vaccine_efficacy=1.0,
            rng=rng,
        )
        results.append(result)

    results = np.stack(results)
    recovery_counts = np.sum(results == states_dict["immune"], axis=(1, 2))
    assert np.any(recovery_counts > 0)
    assert np.any(recovery_counts < np.prod(grid.shape))


def test_disease_rules_stochastic_vaccine(sample_grid_disease):
    """
    Test that immune (vaccinated) cells can become infected probabilistically.

    Parameters
    ----------
    sample_grid_disease : pytest.fixture
        fixture that generates the sample grid
    """
    grid, states_dict = sample_grid_disease
    grid[0, 0] = states_dict["immune"]  # ensure at least one immune cell
    nstates = len(states_dict)
    counts = np.zeros((nstates, 3, 3), dtype=int)
    counts[states_dict["infected"]] = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    rng = np.random.default_rng(123)
    vaccine_efficacy = 0.5
    results = []

    for _ in range(100):
        result = disease_rules(
            grid.copy(),
            neighbour_counts=counts,
            states_dict=states_dict,
            mortality_rate=0.0,
            recovery_rate=0.0,
            infection_rate=1.0,
            vaccine_efficacy=vaccine_efficacy,
            rng=rng,
        )
        results.append(result)

    results = np.stack(results)
    infected_counts = np.sum(results == states_dict["infected"], axis=(1, 2))
    assert np.any(infected_counts > 0)
    assert np.any(infected_counts < np.prod(grid.shape))
