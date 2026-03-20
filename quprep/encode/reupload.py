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
Best for: High-expressivity QNNs. Universal approximation with enough layers.

Reference: Pérez-Salinas et al., Quantum 4, 226 (2020).
"""

from __future__ import annotations

import numpy as np

from quprep.encode.base import BaseEncoder, EncodedResult

_VALID_ROTATIONS = {"ry", "rx", "rz"}


class ReUploadEncoder(BaseEncoder):
    """
    Data re-uploading encoder.

    Parameters
    ----------
    layers : int
        Number of re-upload layers L. Default 3.
    rotation : str
        Rotation gate for data encoding: 'ry', 'rx', or 'rz'.
    """

    def __init__(self, layers: int = 3, rotation: str = "ry"):
        if layers < 1:
            raise ValueError(f"layers must be >= 1, got {layers}.")
        if rotation not in _VALID_ROTATIONS:
            raise ValueError(
                f"Invalid rotation '{rotation}'. Choose from {sorted(_VALID_ROTATIONS)}."
            )
        self.layers = layers
        self.rotation = rotation

    @property
    def n_qubits(self):
        return None  # data-dependent

    @property
    def depth(self):
        return "O(d · layers)"

    def encode(self, x: np.ndarray) -> EncodedResult:
        """
        Encode a 1-D feature vector using data re-uploading.

        The same feature vector is applied `layers` times. Parameters
        store x repeated once — the exporter repeats it `layers` times.
        Normalise input to [−π, π] with 'minmax_pm_pi' before encoding.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or len(x) == 0:
            raise ValueError("ReUploadEncoder.encode() expects a non-empty 1-D array.")

        d = len(x)

        return EncodedResult(
            parameters=x.copy(),
            metadata={
                "encoding": "reupload",
                "n_qubits": d,
                "layers": self.layers,
                "rotation": self.rotation,
                "depth": d * self.layers,
            },
        )
