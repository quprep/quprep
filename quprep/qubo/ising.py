"""QUBO <-> Ising transformations.

QUBO: minimize x^T Q x,  x in {0, 1}^n
Ising: minimize sum_{i<j} J_{ij} s_i s_j + sum_i h_i s_i,  s in {-1, +1}^n

Transformation: s_i = 2*x_i - 1  <=>  x_i = (s_i + 1) / 2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from quprep.qubo.converter import QUBOResult


@dataclass
class IsingResult:
    """
    Ising model representation.

    Attributes
    ----------
    h : np.ndarray, shape (n,)
        Linear (bias) coefficients.
    J : np.ndarray, shape (n, n)
        Quadratic (coupling) coefficients, upper-triangular (J[i,j] for i < j).
    offset : float
        Constant energy offset.
    """

    h: np.ndarray
    J: np.ndarray
    offset: float = 0.0

    def to_qubo(self):
        """
        Convert Ising (h, J) back to QUBO form.

        Uses the inverse transformation x_i = (s_i + 1) / 2:
            Q[i,i] = 2*h_i - 2 * sum_{j != i} J_sym[i,j]
            Q[i,j] = 4 * J[i,j]   for i < j
            offset  = original_offset + sum(J upper-tri) - sum(h)

        Returns
        -------
        QUBOResult
        """
        from quprep.qubo.converter import QUBOResult

        n = len(self.h)
        J_sym = self.J + self.J.T  # symmetric coupling (J is upper-triangular, so J[i,i]=0)

        Q = np.zeros((n, n))
        # Off-diagonal (upper-triangular)
        Q += 4.0 * np.triu(self.J, k=1)
        # Diagonal: 2*h_i - 2 * sum of all couplings involving i
        row_sums = J_sym.sum(axis=1)
        np.fill_diagonal(Q, 2.0 * self.h - 2.0 * row_sums)

        # Constant: offset + sum(J_upper) - sum(h)
        offset = self.offset + np.sum(np.triu(self.J, k=1)) - np.sum(self.h)

        return QUBOResult(Q=Q, offset=offset)

    def __repr__(self) -> str:
        return f"IsingResult(n={len(self.h)}, offset={self.offset:.4f})"


def ising_to_qubo(ising: IsingResult) -> QUBOResult:
    """
    Convert an Ising model back to QUBO form.

    Applies the inverse substitution x_i = (s_i + 1) / 2:

        Q[i,i] = 2*h_i - 2 * sum_{j != i} J_sym[i,j]
        Q[i,j] = 4 * J[i,j]   for i < j
        offset  = original_offset + sum(J_upper) - sum(h)

    Parameters
    ----------
    ising : IsingResult
        Ising model in (h, J) form.

    Returns
    -------
    QUBOResult

    Examples
    --------
    >>> import numpy as np
    >>> from quprep.qubo.problems.maxcut import max_cut
    >>> from quprep.qubo.ising import ising_to_qubo, qubo_to_ising
    >>> adj = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)
    >>> q = max_cut(adj)
    >>> q2 = ising_to_qubo(qubo_to_ising(q))
    >>> np.allclose(q.Q, q2.Q)
    True
    """
    return ising.to_qubo()


def qubo_to_ising(qubo) -> IsingResult:
    """
    Convert a QUBOResult to Ising (h, J) form.

    Uses the substitution x_i = (s_i + 1) / 2:
        J[i,j] = Q[i,j] / 4                          for i < j
        h[i]   = Q[i,i]/2 + sum_{j!=i} Q_sym[i,j]/4
        offset  = original_offset + sum(Q_diag)/2 + sum(Q_upper_off_diag)/4

    Parameters
    ----------
    qubo : QUBOResult
        QUBO problem in upper-triangular form.

    Returns
    -------
    IsingResult
    """
    Q = qubo.Q

    # Symmetric off-diagonal view — diagonal must be zero so row sums are purely off-diagonal
    Q_sym = Q + Q.T - 2.0 * np.diag(np.diag(Q))

    # Coupling matrix (upper-triangular)
    J = np.triu(Q, k=1) / 4.0

    # Bias vector
    off_diag_row_sums = Q_sym.sum(axis=1)  # Q_sym[i,i] = 0 by construction
    h = np.diag(Q) / 2.0 + off_diag_row_sums / 4.0

    # Constant offset
    offset = (
        qubo.offset
        + np.sum(np.diag(Q)) / 2.0
        + np.sum(np.triu(Q, k=1)) / 4.0
    )

    return IsingResult(h=h, J=J, offset=offset)
