# ---------
# IMPORTS
# ---------
from __future__ import annotations

import numpy as np

# -------------------------------------------
# Global kernels (REFACTOR TO UTILS LATER??)
# --------------------------------------------
MOORE_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int)
VON_NEUMANN_KERNEL = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int)

# 3D Moore kernel (all neighbors, excluding self)
MOORE_KERNEL_3D = np.ones((3, 3, 3), dtype=int)
MOORE_KERNEL_3D[1, 1, 1] = 0
