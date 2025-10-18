# ---------
# IMPORTS 
# ---------
import numpy as np

# -------------------------------------------
# Global kernels (REFACTOR TO UTILS LATER??)
# --------------------------------------------
MOORE_KERNEL = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
], dtype=int)

VON_NEUMANN_KERNEL = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=int)
