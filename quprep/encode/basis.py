"""Basis encoding — maps binary/integer features to computational basis states.

Mathematical formulation
------------------------
Given binary x ∈ {0,1}^d:

    |ψ(x)⟩ = |x_1 x_2 ... x_d⟩

Each qubit is set to |0⟩ or |1⟩ via an X gate when x_i = 1.

Properties
----------
Qubits : n = d
Depth  : O(1)
NISQ   : Excellent — only X gates (bit flips), maximally shallow.
Best for: Binary data, integer optimization (QAOA), combinatorial problems.
"""

from __future__ import annotations

import numpy as np

from quprep.encode.base import BaseEncoder, EncodedResult


class BasisEncoder(BaseEncoder):
    """
    Basis (computational state) encoding.

    Input x must be binary {0, 1} per feature. Use quprep.normalize.Scaler('binary').

    Parameters
    ----------
    threshold : float
        Binarization threshold when input is continuous. Default 0.5.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    @property
    def n_qubits(self):
        return None  # data-dependent: n_qubits = n_features

    @property
    def depth(self):
        return 1

    def encode(self, x: np.ndarray) -> EncodedResult:
        raise NotImplementedError("BasisEncoder.encode() — coming in v0.1.0")
