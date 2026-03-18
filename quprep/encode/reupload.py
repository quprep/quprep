"""Data re-uploading encoding — maximum expressivity via repeated feature layers.

Mathematical formulation
------------------------
Given x ∈ [−π, π]^d and L layers:

    |ψ(x)⟩ = U_L(θ_L) S(x) ... U_1(θ_1) S(x) |0⟩^n

where S(x) = ⊗_i R_Y(x_i) is the data-encoding layer and U_l(θ) is a
trainable variational layer. Features are uploaded L times.

Properties
----------
Qubits : n = d
Depth  : O(d · L)
NISQ   : Medium — linear in features and layers.
Best for: High-expressivity quantum neural networks (QNNs). Universal
approximation with sufficient layers.

Reference: Pérez-Salinas et al., "Data re-uploading for a universal
quantum classifier", Quantum 4, 226 (2020).
"""

from __future__ import annotations

import numpy as np

from quprep.encode.base import BaseEncoder, EncodedResult


class ReUploadEncoder(BaseEncoder):
    """
    Data re-uploading encoder.

    Parameters
    ----------
    layers : int
        Number of re-upload layers L. Default 3.
    rotation : str
        Rotation gate used for data encoding: 'ry', 'rx', or 'rz'.
    """

    def __init__(self, layers: int = 3, rotation: str = "ry"):
        self.layers = layers
        self.rotation = rotation

    @property
    def n_qubits(self):
        return None

    @property
    def depth(self):
        return "O(d · layers)"

    def encode(self, x: np.ndarray) -> EncodedResult:
        raise NotImplementedError("ReUploadEncoder.encode() — coming in v0.2.0")
