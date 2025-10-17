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
import dataclasses
from typing import Callable
import numpy as np

@dataclasses.dataclass
class CellularAutomaton:
    """
    Class structure for a very basic, 2 dimensional cellular 
    automaton. This structure defaults to the CGOL rules.
    """
    grid : np.ndarray
    nstates: int
    kernel : np.ndarray
    history: list[np.ndarray] = dataclasses.field(default_factory=list)

    def step(self, rules_fn, convolution_fn):
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
        neighbour_counts = convolution_fn(self.grid, self.kernel, self.nstates)
        self.grid = rules_fn(self.grid, neighbour_counts)
        self.history.append(self.grid.copy())
