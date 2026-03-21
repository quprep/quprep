"""Graph Coloring QUBO formulation.

Graph coloring: assign one of K colors to each node in graph G=(V, E) such
that no two adjacent nodes share the same color.

QUBO uses n*K binary variables x_{i,c}:
    x_{i,c} = 1  =>  node i is assigned color c

Constraints:
    C1 (one color per node): sum_c x[i,c] = 1  for all i
    C2 (valid coloring):     x[i,c] * x[j,c] = 0  for all edges (i,j), all colors c

Both constraints are penalty terms. There is no explicit minimization objective —
any feasible assignment satisfies the graph coloring requirement.

Variable index: v(node, color) = node * n_colors + color

References
----------
Lucas, A. (2014). Ising formulations of many NP problems.
    Frontiers in Physics, 2, 5.
"""

from __future__ import annotations

import numpy as np

from quprep.qubo.converter import QUBOResult


def graph_color(
    adjacency: np.ndarray,
    n_colors: int,
    penalty: float = 10.0,
) -> QUBOResult:
    """
    Build the QUBO for the Graph Coloring problem.

    Parameters
    ----------
    adjacency : np.ndarray, shape (n, n)
        Adjacency matrix of the graph (weighted or 0/1).
        Symmetric; diagonal is ignored.
    n_colors : int
        Number of colors K available for assignment.
    penalty : float
        Lagrange multiplier for both constraints.
        Should be large enough that any constraint violation costs more than
        the best feasible objective. Typically >= n * max_edge_weight.

    Returns
    -------
    QUBOResult
        n * n_colors binary variables. variable_map["x_{i}_{c}"] gives the
        index of variable x_{i,c}. A feasible solution has exactly n variables
        set to 1, one per node.

    Notes
    -----
    A valid K-coloring may not exist for all graphs and K values. If the
    minimum energy solution has non-zero cost, the graph is not K-colorable.
    """
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    W = (adj + adj.T) / 2.0
    np.fill_diagonal(W, 0.0)

    N = n * n_colors
    Q = np.zeros((N, N))
    offset = 0.0

    def idx(node: int, color: int) -> int:
        return node * n_colors + color

    # C1: each node has exactly one color
    # penalty * (sum_c x[i,c] - 1)^2 for each node i
    for i in range(n):
        for c1 in range(n_colors):
            a = idx(i, c1)
            Q[a, a] += penalty * (1.0 - 2.0)   # -penalty per variable
            for c2 in range(c1 + 1, n_colors):
                b = idx(i, c2)
                Q[a, b] += penalty * 2.0
    offset += penalty * n  # n constraints, each contributes penalty * 1^2

    # C2: no two adjacent nodes share a color
    # penalty * x[i,c] * x[j,c] for each edge (i,j) and each color c
    for i in range(n):
        for j in range(i + 1, n):
            w = W[i, j]
            if w == 0.0:
                continue
            for c in range(n_colors):
                a = idx(i, c)
                b = idx(j, c)
                if a > b:
                    a, b = b, a
                Q[a, b] += penalty * w

    var_map = {f"x_{i}_{c}": idx(i, c) for i in range(n) for c in range(n_colors)}
    return QUBOResult(Q=Q, offset=offset, variable_map=var_map)
