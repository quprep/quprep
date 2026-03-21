"""Constraint encoding for QUBO — convert linear/quadratic constraints to penalty terms."""

from __future__ import annotations

import numpy as np


def equality_penalty(A: np.ndarray, b: np.ndarray, penalty: float) -> tuple[np.ndarray, float]:
    """
    Encode linear equality constraints Ax = b as a QUBO penalty matrix.

    Each row of A defines one constraint: a^T x = b_i, encoded as:
        penalty * (a^T x - b_i)^2

    Expanding:
        penalty * (Σ_i a_i^2 x_i + 2 Σ_{i<j} a_i a_j x_i x_j - 2b Σ_i a_i x_i + b^2)

    Which maps to upper-triangular QUBO form:
        Q[i,i] += penalty * (a_i^2 - 2*b*a_i)
        Q[i,j] += penalty * 2 * a_i * a_j   for i < j
        offset  += penalty * b^2

    Parameters
    ----------
    A : np.ndarray, shape (m, n) or (n,)
        Constraint matrix. Each row is one constraint.
        A 1-D array is treated as a single constraint.
    b : np.ndarray, shape (m,) or scalar
        RHS vector. A scalar is broadcast across all constraints.
    penalty : float
        Lagrange multiplier. Must be large enough to enforce constraints;
        a common heuristic is 10x the largest cost coefficient.

    Returns
    -------
    Q_penalty : np.ndarray, shape (n, n)
        Upper-triangular QUBO penalty matrix to add to the cost matrix.
    offset : float
        Constant offset contributed by the penalty terms.
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
    """
    Encode linear inequality constraints Ax <= b as a QUBO penalty matrix.

    Each constraint a^T x <= b_i is converted to an equality by introducing
    binary slack variables z_0, ..., z_{K-1} where K = ceil(log2(max_slack+1)):

        a^T x + sum_k 2^k z_k = b_i

    This expands the variable space from n to n + n_slack, where n_slack is
    the total number of slack bits across all constraints.

    Parameters
    ----------
    A : np.ndarray, shape (m, n) or (n,)
        Constraint matrix. Each row is one constraint.
    b : np.ndarray, shape (m,) or scalar
        RHS values. Must satisfy b_i >= min(a^T x) for feasibility.
    penalty : float
        Lagrange multiplier.

    Returns
    -------
    Q_penalty : np.ndarray, shape (n + n_slack, n + n_slack)
        Upper-triangular QUBO penalty matrix. Rows/columns 0..n-1 correspond
        to original variables; n..n+n_slack-1 are slack variables.
    offset : float
        Constant offset.
    n_slack : int
        Number of slack binary variables added.

    Notes
    -----
    The slack encoding covers slack values 0..2^K-1. If max_slack is not
    a power-of-two minus one, some slack assignments are infeasible but are
    penalised naturally by the equality term. For exact enforcement use a
    large enough penalty (>= max absolute cost coefficient).
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
