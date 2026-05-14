r"""Dense angle encoding — two rotation gates per qubit (Ry + Rz).

Mathematical formulation
------------------------
For d features and n = ⌈d/2⌉ qubits, each qubit k receives two consecutive
features via configurable rotation gates:

$|\psi_k\rangle = R_2(x_{2k+1})\, R_1(x_{2k})\, |0\rangle$

The full state is the tensor product:

$|\psi(x)\rangle = \bigotimes_{k=0}^{n-1} |\psi_k\rangle$

If d is odd, the last qubit receives only the first rotation (second angle = 0).

This halves the qubit count compared to :class:`AngleEncoder` at the cost of
depth 2 (two single-qubit gates per qubit) and no entanglement.

Properties
----------
Qubits : n = ⌈d/2⌉
Depth  : 2
NISQ   : Excellent — depth-2, hardware-native gates, no two-qubit gates.
Best for: Qubit-limited scenarios; encoding paired features (e.g., (r, θ) in polar
          coordinates) onto a single qubit using the full Bloch sphere.
"""

from __future__ import annotations

import math

import numpy as np

from quprep.encode.base import BaseEncoder, EncodedResult

_VALID_ROTATIONS = {"ry", "rx", "rz"}


class DenseAngleEncoder(BaseEncoder):
    """
    Dense angle encoding — 2 features per qubit via two rotation gates.

    Encodes feature pairs as two consecutive single-qubit rotations, using
    ⌈d/2⌉ qubits for d-dimensional input (half the count of
    :class:`~quprep.encode.angle.AngleEncoder`).

    Parameters
    ----------
    first_rotation : str
        First rotation gate applied to each qubit. One of ``'ry'``, ``'rx'``,
        ``'rz'``. Default ``'ry'``.
    second_rotation : str
        Second rotation gate applied to each qubit. One of ``'ry'``, ``'rx'``,
        ``'rz'``. Default ``'rz'``.
    """

    def __init__(self, first_rotation: str = "ry", second_rotation: str = "rz"):
        if first_rotation not in _VALID_ROTATIONS:
            raise ValueError(
                f"first_rotation must be one of {sorted(_VALID_ROTATIONS)}, "
                f"got {first_rotation!r}"
            )
        if second_rotation not in _VALID_ROTATIONS:
            raise ValueError(
                f"second_rotation must be one of {sorted(_VALID_ROTATIONS)}, "
                f"got {second_rotation!r}"
            )
        self.first_rotation = first_rotation
        self.second_rotation = second_rotation

    @property
    def n_qubits(self) -> None:
        return None  # data-dependent: ceil(d/2)

    @property
    def depth(self) -> int:
        return 2

    def encode(self, x: np.ndarray) -> EncodedResult:
        r"""
        Encode a feature vector using two rotation gates per qubit.

        Parameters
        ----------
        x : np.ndarray, shape (d,)
            Normalized feature vector. Recommended normalization:
            ``Scaler('minmax')`` to $[0, \pi]$ for Ry/Rx, or
            ``Scaler('minmax_pm_pi')`` to $[-\pi, \pi]$ for Rz.

        Returns
        -------
        EncodedResult
            ``parameters`` is a flat array of interleaved angles
            ``[r1_0, r2_0, r1_1, r2_1, ...]`` of length ``2 * ⌈d/2⌉``.
            ``metadata`` includes ``encoding``, ``first_rotation``,
            ``second_rotation``, ``n_qubits``, ``depth``, and ``n_features``.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or len(x) == 0:
            raise ValueError("DenseAngleEncoder.encode() expects a non-empty 1-D array.")

        d = len(x)
        n_qubits = math.ceil(d / 2)

        if d % 2 == 1:
            x = np.append(x, 0.0)

        # Interleaved layout: [r1_0, r2_0, r1_1, r2_1, ...]
        parameters = np.empty(2 * n_qubits, dtype=float)
        parameters[0::2] = x[0::2]  # first-rotation angles
        parameters[1::2] = x[1::2]  # second-rotation angles

        return EncodedResult(
            parameters=parameters,
            metadata={
                "encoding": "dense_angle",
                "first_rotation": self.first_rotation,
                "second_rotation": self.second_rotation,
                "n_qubits": n_qubits,
                "depth": 2,
                "n_features": d,
            },
        )
