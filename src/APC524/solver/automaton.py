# ---------------------------------------------------
# Class for basic CA
# ---------------------------------------------------

# ---------
# IMPORTS
# ---------
from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np


@dataclasses.dataclass
class CellularAutomaton:
    """
    Class structure for a very basic, 2 dimensional cellular
    automaton. This structure defaults to the CGOL rules.
    """

    grid: np.ndarray | None = None
    kernel: np.ndarray | None = None
    history: list[np.ndarray] = dataclasses.field(default_factory=list)
    states_dict: dict[str, int] | None = None
    nstates: int = 0

    @classmethod
    def from_init(cls, init_func: Callable[..., CellularAutomaton], **kwargs):
        """
        Factory constructor using a user-provided init function.
        The init function returns a fully initialized CellularAutomaton object.
        Parameters
        ----------
        cls : CellularAutomaton
            the cellular automaton class
        init_func : Callable
            Function which initializes the CellularAutomaton object
        kwargs : dict
            Keyword arguments to pass to the init function
        Returns
        -------
        CellularAutomaton
            Initialized CellularAutomaton object
        """
        return init_func(**kwargs)

    def step(self, rules_fn, convolution_fn, **kwargs):
        """
        Function iterates the game in time.

        Parameters
        ----------
        rules_fn : Callable
            Function which which defines the rules and creates a new grid
        convolution_fn : Callable
            Function which dictates how to apply the kernel search
            to the grid
        kwargs : dict
            Additional keyword arguments to pass to the rules function
        """
        if self.grid is None:
            grid_err = "Grid has not been initialized."
            raise ValueError(grid_err)

        if self.kernel is None:
            kernel_err = "Kernel must be specified to step forward"
            raise ValueError(kernel_err)

        neighbour_counts = convolution_fn(self.grid, self.kernel, self.nstates)
        self.grid = rules_fn(
            self.grid, neighbour_counts, states_dict=self.states_dict, **kwargs
        )
        self.history.append(self.grid.copy())
