# ---------------------------------------------------
# Class for basic CA
# ---------------------------------------------------

# NOTE: I guess the goal would be to inherit from here so that we
# can do some kind of specific class for a specific initialization
# (e.g. we can default init a disease spread scenario or a translation to arduino)
# but also want the user to be able to define their own CA within our rules

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
        """
        return init_func(**kwargs)

    def step(self, rules_fn, convolution_fn, **kwargs):
        """
        Function iterates the game in time.

        NOTE: right now I am keeping the convolution function
        outside of the class so that we can have a CA that is
        3+ dimensional (so that we could do some like simple diffusion
        model maybe in the atmosphere?)

        Options here - make a CellularAutomaton2D and CellularAutomaton3D
        separately and make the convolve_neighbours function a class
        method OR leave it outside and keep the base class multidimensional?

        Parameters
        ----------
        self : CellularAutomaton
            the cellular automaton object
        rules_fn : Callable
            the function which defines the rules and creates a new grid
        convolution_fn : Callable
            the function which dictates how to apply the kernel search
            to the grid
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
