r"""Basis encoding — maps binary/integer features to computational basis states.

Mathematical formulation
------------------------
Given binary $x \in \{0,1\}^d$:

$|\psi(x)\rangle = |x_1\, x_2\, \cdots\, x_d\rangle$

Each qubit is set to $|0\rangle$ or $|1\rangle$ via an X gate when $x_i = 1$.

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
        """
        Binarize x and encode as computational basis state.

        Parameters
        ----------
        x : np.ndarray, shape (d,)
            Feature vector. Values >= threshold → 1; values < threshold → 0.

        Returns
        -------
        EncodedResult
            ``parameters`` = binary vector of 0.0/1.0 values (one per qubit).
            ``metadata`` includes ``encoding``, ``threshold``, ``n_qubits``, ``depth``.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValueError(f"Expected 1D input, got shape {x.shape}")
        if len(x) == 0:
            raise ValueError("Input vector must not be empty")

        bits = (x >= self.threshold).astype(float)
        return EncodedResult(
            parameters=bits,
            metadata={
                "encoding": "basis",
                "threshold": self.threshold,
                "n_qubits": len(bits),
                "depth": 1,
            },
        )
