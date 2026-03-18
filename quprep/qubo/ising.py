"""QUBO ↔ Ising transformations.

QUBO: minimize x^T Q x,  x ∈ {0, 1}^n
Ising: minimize Σ_{i<j} J_{ij} s_i s_j + Σ_i h_i s_i,  s ∈ {−1, +1}^n

Transformation: s_i = 2x_i − 1  ⟺  x_i = (s_i + 1) / 2
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class IsingResult:
    """
    Ising model representation.

    Attributes
    ----------
    h : np.ndarray
        Linear (bias) coefficients, shape (n,).
    J : np.ndarray
        Quadratic (coupling) coefficients, upper-triangular, shape (n, n).
    offset : float
    """

    h: np.ndarray
    J: np.ndarray
    offset: float = 0.0

    def to_qubo(self):
        """Convert back to QUBO form."""
        from quprep.qubo.converter import QUBOResult
        raise NotImplementedError("IsingResult.to_qubo() — coming in v0.3.0")


def qubo_to_ising(qubo) -> IsingResult:
    """Convert a QUBOResult to Ising (h, J) form."""
    raise NotImplementedError("qubo_to_ising() — coming in v0.3.0")
