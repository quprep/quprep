r"""Constraint encoding for QUBO — convert linear/quadratic constraints to penalty terms."""

from __future__ import annotations

import numpy as np


def equality_penalty(A: np.ndarray, b: np.ndarray, penalty: float) -> tuple[np.ndarray, float]:
    r"""
    Encode linear equality constraints $Ax = b$ as a QUBO penalty matrix.

    Each row of $A$ defines one constraint $a^T x = b_i$, penalised as:

    $$\lambda (a^T x - b_i)^2$$

    Expanding and collecting QUBO terms:

    $$
    Q_{ii} \mathrel{+}= \lambda(a_i^2 - 2b_i a_i), \quad
    Q_{ij} \mathrel{+}= 2\lambda a_i a_j \; (i<j), \quad
    \text{offset} \mathrel{+}= \lambda b_i^2
    $$

    Parameters
    ----------
    A : np.ndarray, shape (m, n) or (n,)
        Constraint matrix. Each row is one constraint.
        A 1-D array is treated as a single constraint.
    b : np.ndarray, shape (m,) or scalar
        RHS vector. A scalar is broadcast across all constraints.
    penalty : float
        Lagrange multiplier $\lambda$. Must be large enough to enforce
        constraints; a common heuristic is 10x the largest cost coefficient.

    Returns
    -------
    Q_penalty : np.ndarray, shape (n, n)
        Upper-triangular QUBO penalty matrix to add to the cost matrix.
    offset : float
        Constant offset contributed by the penalty terms.

    Raises
    ------
    ValueError
        If the number of rows in ``A`` does not match the length of ``b``.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    if b.shape[0] == 1:
        b = np.broadcast_to(b, (A.shape[0],))

    m, n = A.shape
    if b.shape[0] != m:
        raise ValueError(f"A has {m} rows but b has {b.shape[0]} elements.")

    Q = np.zeros((n, n))
    offset = 0.0

    for k in range(m):
        a = A[k]
        bk = b[k]
        # Diagonal: penalty * (a_i^2 - 2*bk*a_i)
        np.fill_diagonal(Q, Q.diagonal() + penalty * (a ** 2 - 2.0 * bk * a))
        # Off-diagonal (upper-triangular): penalty * 2 * a_i * a_j
        outer = np.outer(a, a)
        Q += penalty * 2.0 * np.triu(outer, k=1)
        # Constant
        offset += penalty * bk ** 2

    return Q, offset


def inequality_penalty(
    A: np.ndarray,
    b: np.ndarray,
    penalty: float,
) -> tuple[np.ndarray, float, int]:
    r"""
    Encode linear inequality constraints $Ax \leq b$ as a QUBO penalty matrix.

    Each constraint $a^T x \leq b_i$ is converted to an equality by introducing
    $K = \lceil \log_2(\text{max\_slack}+1) \rceil$ binary slack variables
    $z_0, \ldots, z_{K-1}$:

    $$a^T x + \sum_{k=0}^{K-1} 2^k z_k = b_i$$

    This expands the variable space from $n$ to $n + n_{\text{slack}}$.

    Parameters
    ----------
    A : np.ndarray, shape (m, n) or (n,)
        Constraint matrix. Each row is one constraint.
    b : np.ndarray, shape (m,) or scalar
        RHS values. Must satisfy $b_i \geq \min(a^T x)$ for feasibility.
    penalty : float
        Lagrange multiplier $\lambda$.

    Returns
    -------
    Q_penalty : np.ndarray, shape (n + n_slack, n + n_slack)
        Upper-triangular QUBO penalty matrix. Rows/columns $0 \ldots n-1$
        correspond to original variables; $n \ldots n+n_{\text{slack}}-1$
        are slack variables.
    offset : float
        Constant offset.
    n_slack : int
        Number of slack binary variables added.

    Raises
    ------
    ValueError
        If the number of rows in ``A`` does not match the length of ``b``,
        or if any constraint is detected as infeasible
        (``b_i < min(a^T x)``).

    Notes
    -----
    The slack encoding covers slack values $0 \ldots 2^K - 1$. If
    $\text{max\_slack}$ is not a power-of-two minus one, some slack assignments
    are infeasible but are penalised naturally by the equality term.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    if b.shape[0] == 1:
        b = np.broadcast_to(b, (A.shape[0],))

    m, n = A.shape
    if b.shape[0] != m:
        raise ValueError(f"A has {m} rows but b has {b.shape[0]} elements.")

    # Determine slack bits needed per constraint
    slack_specs: list[tuple[int, int, list[int]]] = []  # (start, K, powers)
    total_slack = 0
    for k in range(m):
        a = A[k]
        bk = b[k]
        # Minimum possible value of a^T x (binary x)
        min_val = float(np.sum(a[a < 0]))
        max_slack = int(np.ceil(bk - min_val))
        if max_slack < 0:
            raise ValueError(
                f"Constraint {k} appears infeasible: "
                f"b={bk:.4f} < min(a^T x)={min_val:.4f}."
            )
        K = max(1, int(np.ceil(np.log2(max_slack + 1 + 1e-10))))
        powers = [2 ** j for j in range(K)]
        slack_specs.append((total_slack, K, powers))
        total_slack += K

    n_slack = total_slack
    N = n + n_slack
    Q = np.zeros((N, N))
    offset = 0.0

    for k in range(m):
        a = A[k]
        bk = b[k]
        start, K, powers = slack_specs[k]

        # Augmented constraint: [a | 2^0 2^1 ... 2^{K-1}] * [x | z] = b
        a_aug = np.zeros(N)
        a_aug[:n] = a
        for j, p in enumerate(powers):
            a_aug[n + start + j] = float(p)

        Q_pen, off = equality_penalty(a_aug.reshape(1, -1), np.array([bk]), penalty)
        Q += Q_pen
        offset += off

    return Q, offset, n_slack
