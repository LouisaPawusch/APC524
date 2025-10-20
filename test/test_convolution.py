from __future__ import annotations

import numpy as np
import pytest

from APC524.solver.kernels import MOORE_KERNEL, VON_NEUMANN_KERNEL
from APC524.solver.utils import convolve_neighbours_2D


def test_convolve_neighbours():
    grid = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 0]])
    expected_counts_for_state_1 = np.array([[3, 2, 2], [2, 2, 2], [2, 2, 1]])
    neighbour_counts = convolve_neighbours_2D(grid, MOORE_KERNEL, nstates=2)
    np.testing.assert_array_equal(neighbour_counts[1], expected_counts_for_state_1)


@pytest.fixture
def sample_grid_2_states():
    """
    Creates a sample 3 x 3 grid for testing the convolution function on
    a sample grid with two states for basic CGOL. In this grid, dead is
    0 and alive is 1.

    """
    return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int)


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
        expected_counts = np.array(
            [
                # counts for dead state (0)
                [[2, 3, 2], [3, 6, 3], [2, 3, 2]],
                # counts for alive state (1)
                [[1, 2, 1], [2, 2, 2], [1, 2, 1]],
            ],
            dtype=int,
        )

    else:
        # assumes Von Neumann, maybe we could do this better?
        expected_counts = np.array(
            [[[2, 1, 2], [1, 4, 1], [2, 1, 2]], [[0, 2, 0], [2, 0, 2], [0, 2, 0]]]
        )

    assert np.array_equal(neighbour_counts, expected_counts)
