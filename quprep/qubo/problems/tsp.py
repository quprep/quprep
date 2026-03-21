"""Travelling Salesman Problem (TSP) QUBO formulation.

TSP: given n cities with distance matrix D, find the shortest Hamiltonian
cycle visiting every city exactly once.

QUBO formulation uses n^2 binary variables x_{i,t}:
    x_{i,t} = 1  =>  city i is visited at time step t

Objective (minimize total distance):
    sum_{i,j,t} D[i,j] * x[i,t] * x[j,(t+1) mod n]

Constraints:
    C1: each city visited exactly once:   sum_t x[i,t] = 1  for all i
    C2: each time slot has one city:      sum_i x[i,t] = 1  for all t

Both constraints are enforced with quadratic penalty terms.

Variable index: v(i, t) = i * n + t   (i = city, t = time step)

References
----------
Lucas, A. (2014). Ising formulations of many NP problems.
    Frontiers in Physics, 2, 5.
"""

from __future__ import annotations

import numpy as np

from quprep.qubo.converter import QUBOResult


def tsp(distance_matrix: np.ndarray, penalty: float | None = None) -> QUBOResult:
    """
    Build the QUBO for the Travelling Salesman Problem.

    Parameters
    ----------
    distance_matrix : np.ndarray, shape (n, n)
        Pairwise distances between cities. Asymmetric matrices are supported.
        Self-distances (diagonal) are ignored.
    penalty : float, optional
        Lagrange multiplier for both city and time-slot constraints.
        Defaults to max(D) * n, which is typically large enough to enforce
        feasibility while keeping constraint violations expensive.

    Returns
    -------
    QUBOResult
        n^2 binary variables. variable_map["x_i_t"] gives the index of x_{i,t}.
        A feasible solution has exactly n variables equal to 1, one per city
        and one per time step.
    """
    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]
    if D.shape != (n, n):
        raise ValueError(f"distance_matrix must be square, got {D.shape}.")

    if penalty is None:
        penalty = float(np.max(D)) * n + 1.0

    N = n * n  # total variables
    Q = np.zeros((N, N))
    offset = 0.0

    def idx(city: int, step: int) -> int:
        return city * n + step

    # ---- Objective: minimize total tour distance ----
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = D[i, j]
            for t in range(n):
                t_next = (t + 1) % n
                a = idx(i, t)
                b = idx(j, t_next)
                if a < b:
                    Q[a, b] += d
                elif a > b:
                    Q[b, a] += d
                else:
                    Q[a, a] += d  # shouldn't happen when i != j

    # ---- Constraint 1: each city visited exactly once ----
    # penalty * (sum_t x[i,t] - 1)^2  for each city i
    for i in range(n):
        for t1 in range(n):
            a = idx(i, t1)
            Q[a, a] += penalty * (1.0 - 2.0)  # diagonal: penalty*(1 - 2*1)
            for t2 in range(t1 + 1, n):
                b = idx(i, t2)
                Q[a, b] += penalty * 2.0
    offset += penalty * n  # n cities x penalty * 1^2

    # ---- Constraint 2: each time slot has exactly one city ----
    # penalty * (sum_i x[i,t] - 1)^2  for each time step t
    for t in range(n):
        for i1 in range(n):
            a = idx(i1, t)
            Q[a, a] += penalty * (1.0 - 2.0)
            for i2 in range(i1 + 1, n):
                b = idx(i2, t)
                Q[a, b] += penalty * 2.0
    offset += penalty * n  # n time slots x penalty * 1^2

    var_map = {f"x_{i}_{t}": idx(i, t) for i in range(n) for t in range(n)}
    return QUBOResult(Q=Q, offset=offset, variable_map=var_map)
