r"""Max-Cut QUBO formulation.

Max-Cut: given a weighted graph G=(V, E), partition V into two sets S and S_bar
to maximize the total weight of edges crossing the cut.

QUBO formulation (minimization):

$$\min -\sum_{(i,j) \in E} w_{ij}(x_i + x_j - 2x_ix_j)$$

which maps to: $Q_{ii} -= \sum_j w_{ij}$, $Q_{ij} += 2w_{ij}$ for $i < j$.

References
----------
Lucas, A. (2014). Ising formulations of many NP problems.
    *Frontiers in Physics*, 2, 5. [doi:10.3389/fphy.2014.00005](https://doi.org/10.3389/fphy.2014.00005){target="_blank"}
"""

from __future__ import annotations

import numpy as np

from quprep.qubo.converter import QUBOResult


def max_cut(adjacency: np.ndarray) -> QUBOResult:
    """
    Build the QUBO for the Max-Cut problem.

    Parameters
    ----------
    adjacency : np.ndarray, shape (n, n)
        Weighted adjacency matrix of the graph.
        Symmetric; self-loops (diagonal) are ignored.
        For unweighted graphs, use a 0/1 matrix.

    Returns
    -------
    QUBOResult
        Variable i=0..n-1 is 1 if node i is in partition S, 0 otherwise.
        Minimizing the QUBO objective maximises the cut weight.
    """
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    # Symmetrize in case input is directed
    W = (adj + adj.T) / 2.0
    np.fill_diagonal(W, 0.0)

    Q = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            w = W[i, j]
            if w != 0.0:
                Q[i, j] += 2.0 * w
                Q[i, i] -= w
                Q[j, j] -= w

    var_map = {f"x{i}": i for i in range(n)}
    return QUBOResult(Q=Q, offset=0.0, variable_map=var_map)
