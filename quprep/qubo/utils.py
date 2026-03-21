"""QUBO utility functions — composition, I/O helpers."""

from __future__ import annotations

import numpy as np


def add_qubo(q1, q2, weight: float = 1.0):
    """
    Add two QUBO problems of the same size.

    Useful for multi-objective problems where you want to combine an
    objective term (e.g. max_cut) with a constraint term (e.g. equality
    penalty) that was built separately.

    Parameters
    ----------
    q1 : QUBOResult
        First QUBO.
    q2 : QUBOResult
        Second QUBO. Must have the same number of variables as q1.
    weight : float
        Scalar multiplier applied to q2 before addition. Default is 1.0.

    Returns
    -------
    QUBOResult

    Raises
    ------
    ValueError
        If q1 and q2 have different Q matrix shapes.

    Examples
    --------
    >>> from quprep.qubo.problems.maxcut import max_cut
    >>> from quprep.qubo.constraints import equality_penalty
    >>> from quprep.qubo.converter import QUBOResult
    >>> from quprep.qubo.utils import add_qubo
    >>> import numpy as np
    >>> q_cut = max_cut(np.array([[0,1],[1,0]], dtype=float))
    >>> Q_pen, off = equality_penalty(np.array([[1.0, 1.0]]), np.array([1.0]), 5.0)
    >>> q_pen = QUBOResult(Q=Q_pen, offset=off)
    >>> combined = add_qubo(q_cut, q_pen)
    """
    from quprep.qubo.converter import QUBOResult

    if q1.Q.shape != q2.Q.shape:
        raise ValueError(
            f"Q shapes must match: {q1.Q.shape} vs {q2.Q.shape}. "
            "Use to_qubo(cost, constraints=[...]) to build combined QUBOs directly."
        )

    Q = q1.Q + weight * q2.Q
    offset = q1.offset + weight * q2.offset
    n_original = q1.n_original

    # Merge variable maps — q1 takes precedence on conflict
    var_map = {**q2.variable_map, **q1.variable_map}

    return QUBOResult(Q=Q, offset=offset, variable_map=var_map, n_original=n_original)
