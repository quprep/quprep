r"""ZZ feature map encoding (Havlíček et al., 2019).

Mathematical formulation
------------------------
Input $x \in [0, 2\pi]^d$. For each repetition:

$U_\Phi(x) = U_{ZZ}(x) \cdot H^{\otimes d}$

with

$U_{ZZ}(x) = \exp\!\left(i \sum_i (\pi - x_i) Z_i\right)
              \cdot \exp\!\left(i \sum_{i<j} (\pi - x_i)(\pi - x_j) Z_i Z_j\right)$

Single-qubit terms use $\text{Rz}(2(\pi - x_i))$; pairwise ZZ terms are
implemented as CNOT – $\text{Rz}(2(\pi - x_i)(\pi - x_j))$ – CNOT.

This is the default Qiskit ``ZZFeatureMap`` convention.

Properties
----------
Qubits : n = d
Depth  : O(d² · reps)
NISQ   : Medium — same as IQP; d²·reps two-qubit gates required.
Best for: Quantum kernel methods; drop-in replacement for Qiskit
          ``ZZFeatureMap`` when targeting hardware backends.

Reference: Havlíček et al., *Nature* 567, 209–212 (2019).
"""

from __future__ import annotations

import numpy as np

from quprep.encode.base import BaseEncoder, EncodedResult


class ZZFeatureMapEncoder(BaseEncoder):
    """
    ZZ feature map encoder (Qiskit-compatible convention).

    Implements the feature map from Havlíček et al. (2019) using
    the same parameter convention as Qiskit's ``ZZFeatureMap``:
    single-qubit angles are ``2(π − xᵢ)`` and pairwise angles are
    ``2(π − xᵢ)(π − xⱼ)``.

    Parameters
    ----------
    reps : int
        Number of repetitions of the feature map layer. Default 2.
    """

    def __init__(self, reps: int = 2):
        if reps < 1:
            raise ValueError(f"reps must be >= 1, got {reps}.")
        self.reps = reps

    @property
    def n_qubits(self):
        return None  # data-dependent: n_qubits = n_features

    @property
    def depth(self):
        return "O(d² · reps)"

    def encode(self, x: np.ndarray) -> EncodedResult:
        r"""
        Encode a 1-D feature vector using the ZZ feature map.

        Parameters
        ----------
        x : np.ndarray, shape (d,)
            Normalized feature vector in $[0, 2\pi]$.
            Use ``Scaler('minmax')`` scaled to ``[0, 2π]``.

        Returns
        -------
        EncodedResult
            ``parameters`` = original ``x`` (kept for QASM generation).
            ``metadata`` includes ``encoding``, ``n_qubits``, ``reps``,
            ``depth``, ``single_angles``, ``pair_angles``, ``pairs``.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or len(x) == 0:
            raise ValueError("ZZFeatureMapEncoder.encode() expects a non-empty 1-D array.")

        d = len(x)
        single_angles = 2.0 * (np.pi - x)

        pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
        pair_angles = np.array(
            [2.0 * (np.pi - x[i]) * (np.pi - x[j]) for i, j in pairs],
            dtype=float,
        )

        return EncodedResult(
            parameters=x.copy(),
            metadata={
                "encoding": "zz_feature_map",
                "n_qubits": d,
                "reps": self.reps,
                "depth": d * d * self.reps,
                "single_angles": single_angles.tolist(),
                "pair_angles": pair_angles.tolist(),
                "pairs": pairs,
            },
        )
