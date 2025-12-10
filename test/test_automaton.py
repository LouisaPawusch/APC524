from __future__ import annotations

import numpy as np
import pytest

from APC524.solver.automaton import CellularAutomaton
from APC524.solver.kernels import MOORE_KERNEL, VON_NEUMANN_KERNEL
from APC524.solver.rules import CGOL_rules
from APC524.solver.utils import convolve_neighbours_2D


@pytest.fixture
def sample_grid_2_states():
    """
    Creates a sample 3 x 3 grid for testing the convolution function on
    a sample grid with two states for basic CGOL. In this grid, dead is
    0 and alive is 1.

    """
    grid = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int)
    nstates = 2
    states_dict = {"dead": 0, "alive": 1}
    return grid, nstates, states_dict


@pytest.mark.parametrize("kernel", [MOORE_KERNEL, VON_NEUMANN_KERNEL])
def test_CA_step_with_both_kernels(sample_grid_2_states, kernel):
    """
    Test checks whether after stepping, the grid cells change and the
    and add to the history (verifies step, rules are being called)

    Parameters
    ----------
    sample_grid_2_states : np.ndarray
        the sample grid generated in the fixture
    kernel : np.ndarray
        the sample kernel generated in the fixture (iterates search
        over the Moore and Von Neumann neighbourhoods)
    """
    grid, nstates, states_dict = sample_grid_2_states

    ca = CellularAutomaton(
        grid=grid.copy(), nstates=nstates, kernel=kernel, states_dict=states_dict
    )

    before = ca.grid.copy()
    ca.step(CGOL_rules, convolve_neighbours_2D)
    after = ca.grid

    assert len(ca.history) == 1
    assert not np.array_equal(before, after)
