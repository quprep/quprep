"""Number Partitioning QUBO formulation.

Number partitioning: given n positive numbers v_i, split them into two
subsets A and B such that the difference |sum(A) - sum(B)| is minimised
(ideally zero for a perfect partition).

Let x_i = 1 if value i is in subset A, 0 if in subset B.
Define s_i = 2*x_i - 1 in {-1, +1}. A perfect partition requires:

    sum_i v_i * s_i = 0

The objective to minimize is the squared difference:

    (sum_i v_i * s_i)^2 = (sum_i v_i * (2*x_i - 1))^2

Expanding and collecting QUBO terms (using x_i^2 = x_i for binary x):
    S = sum_i v_i

    Q[i,i] = v_i * (v_i - S) * 4 / 4    ... (simplified)
    Q[i,j] = 2 * v_i * v_j  for i < j
    offset  = S^2

Note: the penalty argument scales the entire objective — useful when
combining with other QUBO terms via add_qubo().

References
----------
Lucas, A. (2014). Ising formulations of many NP problems.
    Frontiers in Physics, 2, 5.
"""

from __future__ import annotations

import numpy as np

from quprep.qubo.converter import QUBOResult


def number_partition(values: np.ndarray, penalty: float = 1.0) -> QUBOResult:
    """
    Build the QUBO for the Number Partitioning problem.

    Parameters
    ----------
    values : np.ndarray, shape (n,)
        Positive numbers to partition.
    penalty : float
        Global scale factor. Set > 1 when combining with other QUBO objectives
        to ensure the partition constraint dominates. Default is 1.0.

    Returns
    -------
    QUBOResult
        Variable x_i = 1 means value i goes into subset A.
        Minimum energy = 0 indicates a perfect partition exists.
        Minimum energy > 0 means no perfect partition; the solution with
        lowest energy is the most balanced achievable split.

    Examples
    --------
    >>> import numpy as np
    >>> from quprep.qubo.problems.number_partition import number_partition
    >>> from quprep.qubo.solver import solve_brute
    >>> v = np.array([3.0, 1.0, 1.0, 2.0, 2.0, 1.0])  # sum=10, perfect split at 5
    >>> sol = solve_brute(number_partition(v))
    >>> sol.energy   # should be 0.0 for a perfect partition
    0.0
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    S = float(np.sum(v))

    Q = np.zeros((n, n))
    offset = 0.0

    # (sum_i v_i s_i)^2 where s_i = 2x_i - 1
    # = (2 sum_i v_i x_i - S)^2
    # = 4 (sum_i v_i x_i)^2 - 4S sum_i v_i x_i + S^2
    #
    # Expanding (sum_i v_i x_i)^2:
    #   = sum_i v_i^2 x_i^2 + 2 sum_{i<j} v_i v_j x_i x_j
    #   = sum_i v_i^2 x_i   + 2 sum_{i<j} v_i v_j x_i x_j   (binary)
    #
    # Diagonal: 4 v_i^2 - 4S v_i = 4 v_i (v_i - S)
    np.fill_diagonal(Q, Q.diagonal() + penalty * 4.0 * v * (v - S))

    # Off-diagonal: 4 * 2 * v_i v_j / 2 ... wait let me redo:
    # 4*(sum v_i x_i)^2 = 4*sum_i v_i^2 x_i + 4*2*sum_{i<j} v_i v_j x_i x_j
    #                   = 4 v_i^2 x_i diagonal + 8 v_i v_j x_i x_j off-diag
    # Combined with -4S sum v_i x_i (diagonal only):
    # Q[i,i] = 4v_i^2 - 4S v_i = 4v_i(v_i - S)
    # Q[i,j] = 8 v_i v_j   for i < j
    # offset  = S^2
    #
    # (The fill_diagonal above already set Q[i,i] = penalty * 4 v_i(v_i-S))
    outer = np.outer(v, v)
    Q += penalty * 8.0 * np.triu(outer, k=1)
    offset += penalty * S ** 2

    var_map = {f"x{i}": i for i in range(n)}
    return QUBOResult(Q=Q, offset=offset, variable_map=var_map)
