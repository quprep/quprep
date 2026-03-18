"""Angle encoding — maps features to qubit rotation angles.

Mathematical formulation
------------------------
Given normalized x ∈ [0, π]^d (Ry) or [−π, π]^d (Rx/Rz):

    |ψ(x)⟩ = ⊗_{i=1}^{d} R_G(x_i)|0⟩

where R_G is the chosen rotation gate (Ry, Rx, or Rz).

Properties
----------
Qubits : n = d
Depth  : O(1)
NISQ   : Excellent — shallow, hardware-native gates only.
Best for: Most QML tasks (default recommendation).
"""

from __future__ import annotations

import numpy as np

from quprep.encode.base import BaseEncoder, EncodedResult


class AngleEncoder(BaseEncoder):
    """
    Angle encoding using single-qubit rotation gates.

    Parameters
    ----------
    rotation : str
        Rotation gate to use: 'ry' (default), 'rx', or 'rz'.
    """

    def __init__(self, rotation: str = "ry"):
        if rotation not in ("ry", "rx", "rz"):
            raise ValueError(f"rotation must be 'ry', 'rx', or 'rz', got '{rotation}'")
        self.rotation = rotation

    @property
    def n_qubits(self):
        return None  # data-dependent: n_qubits = n_features

    @property
    def depth(self):
        return 1

    def encode(self, x: np.ndarray) -> EncodedResult:
        """
        Encode feature vector x as rotation angles.

        x must be pre-normalized to [0, π] for 'ry', or [−π, π] for 'rx'/'rz'.
        Use quprep.normalize.auto_normalizer(encoding) to apply correct scaling.
        """
        raise NotImplementedError("AngleEncoder.encode() — coming in v0.1.0")
