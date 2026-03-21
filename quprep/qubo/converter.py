"""Convert cost matrices and constraints to QUBO format."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from quprep.qubo.ising import IsingResult


class QUBOResult:
    """
    Result of a QUBO conversion.

    Attributes
    ----------
    Q : np.ndarray, shape (n, n)
        Upper-triangular QUBO matrix.
        Q[i,i] are linear (bias) terms; Q[i,j] for i<j are quadratic couplings.
        Suitable for D-Wave samplers, QAOA, and OpenQAOA.
    offset : float
        Constant offset term (does not affect the optimal solution).
    variable_map : dict
        Mapping from variable names to matrix indices. Empty by default.
    n_original : int
        Number of original (non-slack) variables. Equals Q.shape[0] unless
        inequality constraints introduced slack variables.
    """

    def __init__(
        self,
        Q: np.ndarray,
        offset: float = 0.0,
        variable_map: dict | None = None,
        n_original: int | None = None,
    ):
        self.Q = Q
        self.offset = offset
        self.variable_map = variable_map or {}
        self.n_original = n_original if n_original is not None else Q.shape[0]

    def to_ising(self) -> IsingResult:
        """Convert QUBO to Ising (h, J) form."""
        from quprep.qubo.ising import qubo_to_ising
        return qubo_to_ising(self)

    def to_dict(self) -> dict:
        """
        Serialize to a plain Python dict (JSON-compatible).

        Returns
        -------
        dict with keys: Q, offset, variable_map, n_original.
        Q is stored as a nested list (use json.dumps to save).
        """
        return {
            "Q": self.Q.tolist(),
            "offset": float(self.offset),
            "variable_map": self.variable_map,
            "n_original": self.n_original,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QUBOResult":
        """
        Deserialize from a dict produced by ``to_dict()``.

        Parameters
        ----------
        d : dict
            Dict with keys Q, offset, variable_map, n_original.

        Returns
        -------
        QUBOResult
        """
        return cls(
            Q=np.array(d["Q"]),
            offset=float(d.get("offset", 0.0)),
            variable_map=d.get("variable_map", {}),
            n_original=d.get("n_original"),
        )

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the QUBO objective for a given binary assignment.

        Computes x^T Q x + offset.

        Parameters
        ----------
        x : array-like, shape (n,)
            Binary assignment vector with values in {0, 1}.

        Returns
        -------
        float
            Objective value including the constant offset.

        Examples
        --------
        >>> import numpy as np
        >>> from quprep.qubo.problems.maxcut import max_cut
        >>> adj = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)
        >>> q = max_cut(adj)
        >>> q.evaluate(np.array([0, 1, 1]))  # cut between node 0 and {1,2}
        -2.0
        """
        x = np.asarray(x, dtype=float)
        return float(x @ self.Q @ x) + self.offset

    def to_dwave(self) -> dict:
        """
        Export QUBO as a D-Wave Ocean SDK-compatible linear/quadratic dict.

        Returns a dict mapping ``(i, j)`` tuples to coefficients, where:

        - ``i == j`` encodes linear (bias) terms (from the diagonal of Q)
        - ``i < j`` encodes quadratic (coupling) terms (from upper triangle of Q)

        Zero entries are omitted. The result can be passed directly to
        ``dimod.BinaryQuadraticModel.from_qubo()`` or
        ``dwave.system.DWaveSampler``.

        Returns
        -------
        dict
            ``{(i, j): float}`` with i <= j.

        Examples
        --------
        >>> import numpy as np
        >>> from quprep.qubo.problems.maxcut import max_cut
        >>> adj = np.array([[0,1],[1,0]], dtype=float)
        >>> max_cut(adj).to_dwave()
        {(0, 0): -1.0, (1, 1): -1.0, (0, 1): 2.0}
        """
        result = {}
        n = self.Q.shape[0]
        for i in range(n):
            for j in range(i, n):
                val = float(self.Q[i, j])
                if abs(val) > 1e-12:
                    result[(i, j)] = val
        return result

    def __repr__(self) -> str:
        n = self.Q.shape[0]
        return f"QUBOResult(n_variables={n}, offset={self.offset:.4f})"


def to_qubo(
    cost_matrix: np.ndarray,
    constraints: list[dict] | None = None,
    penalty: float = 10.0,
) -> QUBOResult:
    """
    Convert a cost matrix and optional linear equality constraints to QUBO.

    The QUBO objective is: minimize x^T Q x,  x in {0, 1}^n

    The input cost_matrix can be any square real matrix. It is converted to
    upper-triangular QUBO form:
        Q[i,i] = M[i,i]                  (linear / bias term)
        Q[i,j] = M[i,j] + M[j,i] i < j  (quadratic coupling)

    Constraint penalties are added on top via the Lagrangian approach.

    Parameters
    ----------
    cost_matrix : np.ndarray, shape (n, n)
        Quadratic cost matrix. Diagonal entries encode linear terms.
    constraints : list of dicts, optional
        Equality and inequality constraints. Each dict must have:
            "A"       : np.ndarray, shape (m, n) or (n,)
            "b"       : np.ndarray or scalar
            "type"    : "eq" (default) or "ineq" (Ax <= b via slack variables)
            "penalty" : float (optional, falls back to global penalty)
        Inequality constraints expand the variable count by adding binary
        slack variables. ``result.n_original`` records the original count.
    penalty : float
        Default Lagrange multiplier. Heuristic: 10x the largest |cost| entry.

    Returns
    -------
    QUBOResult
        ``result.n_original`` = n (original variables).
        If inequality constraints are present, ``result.Q.shape[0] > n``.
    """
    M = np.asarray(cost_matrix, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"cost_matrix must be square 2-D, got shape {M.shape}.")
    n = M.shape[0]

    # Build upper-triangular QUBO from cost matrix
    Q = np.zeros((n, n))
    np.fill_diagonal(Q, np.diag(M))
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    Q[upper_mask] = M[upper_mask] + M.T[upper_mask]

    total_offset = 0.0
    total_slack = 0

    if constraints:
        from quprep.qubo.constraints import equality_penalty, inequality_penalty

        for c in constraints:
            A = np.asarray(c["A"], dtype=float)
            b = np.asarray(c["b"], dtype=float)
            lam = float(c.get("penalty", penalty))
            ctype = c.get("type", "eq")

            if ctype == "ineq":
                # Expand Q to accommodate slack variables
                Q_pen, pen_offset, n_slack = inequality_penalty(A, b, lam)
                # Pad current Q to match expanded size
                new_size = Q.shape[0] + n_slack
                Q_expanded = np.zeros((new_size, new_size))
                Q_expanded[: Q.shape[0], : Q.shape[0]] = Q
                Q = Q_expanded + Q_pen
                total_offset += pen_offset
                total_slack += n_slack
            else:
                # Equality constraint — pad A to current Q size if slack was added
                cur_n = Q.shape[0]
                if A.ndim == 1:
                    A = A.reshape(1, -1)
                if A.shape[1] < cur_n:
                    pad = np.zeros((A.shape[0], cur_n - A.shape[1]))
                    A = np.concatenate([A, pad], axis=1)
                Q_pen, pen_offset = equality_penalty(A, b, lam)
                Q += Q_pen
                total_offset += pen_offset

    return QUBOResult(Q=Q, offset=total_offset, n_original=n)
