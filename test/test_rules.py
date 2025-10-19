#!/usr/bin/envs python3

import pytest
import numpy as np
from APC524.solver.kernels import MOORE_KERNEL, VON_NEUMANN_KERNEL
from APC524.solver.rules import CGOL_rules

#-----------------------------
# Test CGOL Rules 
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
    Tests rule 2: cells with two or more neighbours survive

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
