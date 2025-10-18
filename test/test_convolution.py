#!/usr/bin/envs python3

import numpy as np
from APC524.solver.utils import convolve_neighbours_2D
from APC524.solver.kernels import MOORE_KERNEL

def test_convolve_neighbours():
    grid = np.array([
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 0]
    ])
    expected_counts_for_state_1 = np.array([
        [3, 2, 2],
        [2, 2, 2],
        [2, 2, 1]
    ])
    neighbour_counts = convolve_neighbours_2D(grid, MOORE_KERNEL, nstates=2)
    np.testing.assert_array_equal(neighbour_counts[1], expected_counts_for_state_1)

