r"""Angle encoding — maps features to qubit rotation angles.

Mathematical formulation
------------------------
Given normalized $x \in [0, \pi]^d$ (Ry) or $x \in [-\pi, \pi]^d$ (Rx/Rz):

$|\psi(x)\rangle = \bigotimes_{i=1}^{d} R_G(x_i)|0\rangle$

where $R_G$ is the chosen rotation gate (Ry, Rx, or Rz).

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
        r"""
        Encode feature vector x as rotation angles.

        Parameters
        ----------
        x : np.ndarray, shape (d,)
            Normalized feature vector. Must be in $[0, \pi]$ for 'ry',
            or $[-\pi, \pi]$ for 'rx'/'rz'. Use
            ``quprep.normalize.auto_normalizer(encoding)`` for correct scaling.

        Returns
        -------
        EncodedResult
            ``parameters`` = x (rotation angles, one per qubit).
            ``metadata`` includes ``encoding``, ``rotation``, ``n_qubits``, ``depth``.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValueError(f"Expected 1D input, got shape {x.shape}")
        if len(x) == 0:
            raise ValueError("Input vector must not be empty")
        return EncodedResult(
            parameters=x.copy(),
            metadata={
                "encoding": "angle",
                "rotation": self.rotation,
                "n_qubits": len(x),
                "depth": 1,
            },
        )
