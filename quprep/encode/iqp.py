"""IQP (Instantaneous Quantum Polynomial) encoding.

Mathematical formulation
------------------------
Given x ∈ [−π, π]^d with pairwise products x_i·x_j:

    |ψ(x)⟩ = U_Φ(x) H^⊗n |0⟩^n

where U_Φ(x) = exp(i Σ_i x_i Z_i + i Σ_{i<j} x_i x_j Z_i Z_j)

Properties
----------
Qubits : n = d
Depth  : O(d²) — quadratic in features.
NISQ   : Medium — d² two-qubit gates required.
Best for: Kernel methods with quantum advantage arguments.

Reference: Havlíček et al., Nature 567, 209–212 (2019).
"""

from __future__ import annotations

import numpy as np

from quprep.encode.base import BaseEncoder, EncodedResult


class IQPEncoder(BaseEncoder):
    """
    IQP feature map encoding.

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
        return None  # data-dependent

    @property
    def depth(self):
        return "O(d² · reps)"

    def encode(self, x: np.ndarray) -> EncodedResult:
        """
        Encode a 1-D feature vector using the IQP feature map.

        Parameters store [x_0..x_{d-1}, x_0·x_1, ..., x_{d-2}·x_{d-1}].
        Normalise input to [−π, π] with 'minmax_pm_pi' before encoding.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or len(x) == 0:
            raise ValueError("IQPEncoder.encode() expects a non-empty 1-D array.")

        d = len(x)
        pairs = np.array(
            [x[i] * x[j] for i in range(d) for j in range(i + 1, d)],
            dtype=float,
        )
        parameters = np.concatenate([x, pairs])

        return EncodedResult(
            parameters=parameters,
            metadata={
                "encoding": "iqp",
                "n_qubits": d,
                "reps": self.reps,
                "depth": d * d * self.reps,
                "n_pairs": len(pairs),
            },
        )
