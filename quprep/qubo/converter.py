"""Convert cost matrices and constraints to QUBO format."""

from __future__ import annotations

import numpy as np


class QUBOResult:
    """
    Result of a QUBO conversion.

    Attributes
    ----------
    Q : np.ndarray
        Upper-triangular QUBO matrix, shape (n, n).
        Suitable for D-Wave samplers, QAOA, and OpenQAOA.
    offset : float
        Constant offset term.
    variable_map : dict
        Mapping from variable names to matrix indices.
    """

    def __init__(self, Q: np.ndarray, offset: float = 0.0, variable_map: dict | None = None):
        self.Q = Q
        self.offset = offset
        self.variable_map = variable_map or {}

    def to_ising(self) -> "IsingResult":
        """Convert QUBO to Ising (h, J) form."""
        from quprep.qubo.ising import qubo_to_ising
        return qubo_to_ising(self)

    def __repr__(self) -> str:
        n = self.Q.shape[0]
        return f"QUBOResult(n_variables={n}, offset={self.offset:.4f})"


def to_qubo(cost_matrix: np.ndarray, constraints: list | None = None, penalty: float = 10.0) -> QUBOResult:
    """
    Convert a cost matrix and optional constraints to a QUBO problem.

    Parameters
    ----------
    cost_matrix : np.ndarray
        Quadratic cost matrix. Can be asymmetric; will be symmetrized.
    constraints : list of constraint dicts, optional
        Equality and inequality constraints to encode as quadratic penalties.
    penalty : float
        Lagrange multiplier for constraint penalty terms.

    Returns
    -------
    QUBOResult
    """
    raise NotImplementedError("to_qubo() — coming in v0.3.0")
