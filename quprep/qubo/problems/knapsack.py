"""0/1 Knapsack QUBO formulation.

Knapsack: given n items with values v_i and weights w_i, select a subset
to maximize total value without exceeding capacity W.

QUBO formulation (minimization):
    minimize  -sum_i v_i * x_i  +  penalty * (sum_i w_i * x_i - W)^2

The capacity constraint is enforced via a quadratic penalty term.
The penalty coefficient should be at least max(v_i) to ensure feasibility.

Note
----
This is a penalty-based (soft) formulation. Infeasible solutions have higher
energy, but the exact cut-off depends on penalty strength. For hard enforcement
use slack binary variables (adds ceil(log2(W+1)) ancilla qubits).

References
----------
Lucas, A. (2014). Ising formulations of many NP problems.
    Frontiers in Physics, 2, 5.
"""

from __future__ import annotations

import numpy as np

from quprep.qubo.converter import QUBOResult


def knapsack(
    weights: np.ndarray,
    values: np.ndarray,
    capacity: float,
    penalty: float | None = None,
) -> QUBOResult:
    """
    Build the QUBO for the 0/1 Knapsack problem.

    Parameters
    ----------
    weights : np.ndarray, shape (n,)
        Item weights.
    values : np.ndarray, shape (n,)
        Item values (non-negative).
    capacity : float
        Maximum total weight W.
    penalty : float, optional
        Lagrange multiplier for the capacity constraint.
        Defaults to max(values) + 1, which is tight enough to enforce
        feasibility for integer-valued instances.

    Returns
    -------
    QUBOResult
        Variable x_i = 1 means item i is selected.
    """
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    n = len(w)
    if len(v) != n:
        raise ValueError("weights and values must have the same length.")

    if penalty is None:
        penalty = float(np.max(v)) + 1.0

    Q = np.zeros((n, n))
    offset = 0.0

    # Objective: minimize -v_i * x_i
    np.fill_diagonal(Q, Q.diagonal() - v)

    # Capacity penalty: penalty * (sum w_i x_i - W)^2
    # Diagonal: penalty * (w_i^2 - 2*W*w_i)
    np.fill_diagonal(Q, Q.diagonal() + penalty * (w ** 2 - 2.0 * capacity * w))
    # Off-diagonal: penalty * 2 * w_i * w_j
    outer = np.outer(w, w)
    Q += penalty * 2.0 * np.triu(outer, k=1)
    offset += penalty * capacity ** 2

    var_map = {f"x{i}": i for i in range(n)}
    return QUBOResult(Q=Q, offset=offset, variable_map=var_map)
