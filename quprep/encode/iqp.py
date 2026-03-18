"""IQP (Instantaneous Quantum Polynomial) encoding.

Mathematical formulation
------------------------
Given x ∈ [−π, π]^d with pairwise products x_i·x_j:

    |ψ(x)⟩ = U_Φ(x) H^⊗n |0⟩^n

where U_Φ(x) = exp(i Σ_i x_i Z_i + i Σ_{i<j} x_i x_j Z_i Z_j)

This applies Hadamards, then a diagonal phase encoding using single-qubit
Z rotations (features) and two-qubit ZZ interactions (feature products).

Properties
----------
Qubits : n = d
Depth  : O(d²) — quadratic in features.
NISQ   : Medium — d² two-qubit gates required.
Best for: Kernel methods. Proven quantum advantage for specific problems.

Reference: Havlíček et al., "Supervised learning with quantum-enhanced
feature spaces", Nature 567, 209–212 (2019).
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
        More reps → higher expressivity, greater depth.
    """

    def __init__(self, reps: int = 2):
        self.reps = reps

    @property
    def n_qubits(self):
        return None  # data-dependent: n = d

    @property
    def depth(self):
        return "O(d² · reps)"

    def encode(self, x: np.ndarray) -> EncodedResult:
        raise NotImplementedError("IQPEncoder.encode() — coming in v0.2.0")
