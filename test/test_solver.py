#!/usr/bin/envs python3
"""
test_solver.py

This code tests the helper functions and CA class in src/solver.py

It might be better to refactor everything and have the helper functions
in a separate utils.py module, but we probably need to decide what to do
with the class methods first?

"""
import pytest
import numpy as np
from src.APC524.solver import convolve_neighbours_2D, CGOL_rules, MOORE_KERNEL, VON_NEUMANN_KERNEL, CellularAutomaton

#-----------------------------
# Test 2D Convolution Solver 
# ----------------------------
@pytest.fixture
def sample_grid_2_states():
    """
    Creates a sample 3 x 3 grid for testing the convolution function on
    a sample grid with two states for basic CGOL. In this grid, dead is
    0 and alive is 1.

    """
    return np.array( [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=int)

@pytest.fixture(params=[MOORE_KERNEL, VON_NEUMANN_KERNEL])
def kernel(request):
    """
    Fixture to provide both Moore and Von Neumann kernels.
    The test using this fixture will automatically run once per kernel.
    """
    return request.param

def test_neighbour_counts_2_states(sample_grid_2_states, kernel):
    """
    Checks whether the neighbour counts for each cell are correct based
    on the type of kernel used (Von Neumann or Moore) for a sample grid
    with two states

    NOTE: I'm not sure if this is a bad way to do this and so I am 
    just parameterizing and using fixtures for the sake of parameterizing
    and using fixtures... if it would be better to split up these tests we
    definitely can.

    Parameters
    ----------
    sample_grid_2_states : np.ndarray
        the sample grid generated in the fixture
    kernel : np.ndarray
        the sample kernel generated in the fixture (iterates search
        over the Moore and Von Neumann neighbourhoods)
    """
    grid = sample_grid_2_states
    nstates = 2

    # count all the neighbouring cells with convolution
    neighbour_counts = convolve_neighbours_2D(grid, kernel, nstates)

    # now define the expected counts for each 
    if np.array_equal(kernel, MOORE_KERNEL):
        # expected counts for each cell state - should be a 2D
        # array that shows the number of neighbours in state 0
        # and the number in state 1 for each cell.
        expected_counts = np.array([
            # counts for dead state (0)
            [[2, 3, 2],
             [3, 6, 3],
             [2, 3, 2]], 

            # counts for alive state (1)
             [[1, 2, 1],
              [2, 2, 2],
              [1, 2, 1]]
        ], dtype=int)

    else:
        # assumes Von Neumann, maybe we could do this better?
        expected_counts = np.array([
            [[2, 1, 2],
             [1, 4, 1],
             [2, 1, 2]], 

             [[0, 2, 0],
              [2, 0, 2],
              [0, 2, 0]]
        ])

    assert np.array_equal(neighbour_counts, expected_counts)

# ---------------------------------
# Test the neighbour rules for CGOL
# ----------------------------------

def test_CGOL_rules_underpopulation(sample_grid_2_states):
    """
    Tests rule 1: Cells with less than 2 neighbours die
    of lonliness :(

    Parameters
    ----------
    sample_grid_2_states : np.ndarray
        the original sample grid 

    """
    grid = sample_grid_2_states.copy()
    # make an array with the same shape as the output from the 2D convolution
    counts = np.zeros((2, 3, 3), dtype=int)

    counts[1, 1, 1] = 1  # give the center one living neighbour
    result = CGOL_rules(grid, counts)

    assert result[1, 1] == 0  # be sure center dies of loneliness

def test_CGOL_rules_survival(sample_grid_2_states):
    """
    Tests rule 2: cells with two or more neighbours survive - Yippie :)

    Parameters
    ----------
    sample_grid_2_states : np.ndarray
        the original sample grid 

    """
    grid = sample_grid_2_states.copy()
    counts = np.zeros((2, 3, 3), dtype=int)

    counts[1, 1, 1] = 2
    counts[1, 1, 2] = 3
    result = CGOL_rules(grid, counts)

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
    grid = sample_grid_2_states.copy()
    counts = np.zeros((2, 3, 3), dtype=int)

    counts[1, 1, 1] = 4
    result = CGOL_rules(grid, counts)

    assert result[1, 1] == 0

def test_CGOL_rules_reproduction(sample_grid_2_states):
    """
    Tests rule 4: DEAD cells with exactly 3 live neighbours are reborn

    Parameters
    ----------
    sample_grid_2_states : np.ndarray
        the original sample grid
    """
    grid = sample_grid_2_states.copy()
    grid[1, 1] = 0  # ensure dead

    counts = counts = np.zeros((2, 3, 3), dtype=int)
    counts[1, 1, 1] = 3

    result = CGOL_rules(grid, counts)
    assert result[1, 1] == 1

# ----------------------------
# Test the CA object
# ----------------------------
@pytest.mark.parametrize("kernel", [MOORE_KERNEL, VON_NEUMANN_KERNEL])
def test_CA_step_with_both_kernels(sample_grid_2_states, kernel):
    """
    Test checks whether after stepping, the grid cells change and the 
    and add to the history (verifies step, rules are being called)

    We could probably add a much better explicit test here which actively
    steps through the CA for the initial grid (they all die p fast) and 
    checks those.

    Parameters
    ----------
    sample_grid_2_states : np.ndarray
        the sample grid generated in the fixture
    kernel : np.ndarray
        the sample kernel generated in the fixture (iterates search
        over the Moore and Von Neumann neighbourhoods)
    
    """
    ca = CellularAutomaton(sample_grid_2_states.copy(), nstates=2, kernel=kernel)

    before = ca.grid.copy()
    ca.step(CGOL_rules, convolve_neighbours_2D)
    after = ca.grid

    assert len(ca.history) == 1
    assert not np.array_equal(before, after)
