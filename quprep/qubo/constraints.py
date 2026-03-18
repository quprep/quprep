"""Constraint encoding for QUBO — convert linear/quadratic constraints to penalty terms."""

from __future__ import annotations

import numpy as np


def equality_penalty(A: np.ndarray, b: np.ndarray, penalty: float) -> np.ndarray:
    """
    Encode linear equality constraints Ax = b as QUBO penalty terms.

    Each constraint a^T x = b_i is encoded as:
        penalty * (a^T x - b_i)^2

    Returns a QUBO Q matrix of penalty terms to add to the cost matrix.

    Parameters
    ----------
    A : np.ndarray, shape (m, n)
        Constraint matrix.
    b : np.ndarray, shape (m,)
        RHS vector.
    penalty : float
        Lagrange multiplier. Should be large enough to enforce constraints.

    Returns
    -------
    np.ndarray, shape (n, n)
    """
    raise NotImplementedError("equality_penalty() — coming in v0.3.0")
