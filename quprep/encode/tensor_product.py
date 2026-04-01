r"""Tensor product encoding — full Bloch sphere parameterization.

Mathematical formulation
------------------------
Each qubit $k$ is encoded using two consecutive features
$(x_{2k}, x_{2k+1})$ as a pure state on the Bloch sphere:

$|\psi_k\rangle = R_y(x_{2k}) R_z(x_{2k+1}) |0\rangle$

The full state is the tensor product:

$|\psi(x)\rangle = \bigotimes_{k=0}^{n-1} |\psi_k\rangle$

where $n = \lfloor d/2 \rfloor$ (if $d$ is odd the last feature
is encoded using $R_y$ only with $R_z(0)$).

This encoding uses the full Bloch sphere parametrization, giving more
expressive single-qubit states than plain angle encoding at the cost
of two features per qubit.

Properties
----------
Qubits : n = ⌈d/2⌉
Depth  : 2  (one Ry + one Rz per qubit, no entanglement)
NISQ   : Excellent — depth-2, hardware-native gates, no two-qubit gates.
Best for: Structured data where pairs of features have natural
          geometric meaning; compact encoding of high-dimensional data.
"""

from __future__ import annotations

import math

import numpy as np

from quprep.encode.base import BaseEncoder, EncodedResult


class TensorProductEncoder(BaseEncoder):
    """
    Tensor product encoding using full Bloch sphere parameterization.

    Encodes pairs of features as Ry + Rz rotations on individual
    qubits with no entanglement.  Uses ``⌈d/2⌉`` qubits for
    ``d``-dimensional input.

    No parameters.
    """

    @property
    def n_qubits(self):
        return None  # data-dependent: ceil(d/2)

    @property
    def depth(self):
        return 2

    def encode(self, x: np.ndarray) -> EncodedResult:
        r"""
        Encode a 1-D feature vector onto the Bloch sphere.

        Parameters
        ----------
        x : np.ndarray, shape (d,)
            Normalized feature vector. Use ``Scaler('minmax')`` scaled to
            $[0, \pi]$ for $R_y$ angles (polar) and ``Scaler('minmax_pm_pi')``
            scaled to $[-\pi, \pi]$ for $R_z$ angles (azimuthal).
            In practice a single ``Scaler('minmax')`` applied to all features
            works well as a starting point.

        Returns
        -------
        EncodedResult
            ``parameters`` = flat array of alternating Ry/Rz angles,
            length ``2 * ⌈d/2⌉`` (odd inputs are zero-padded for the
            missing Rz angle).
            ``metadata`` includes ``encoding``, ``n_qubits``, ``depth``,
            ``ry_angles``, ``rz_angles``.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or len(x) == 0:
            raise ValueError("TensorProductEncoder.encode() expects a non-empty 1-D array.")

        d = len(x)
        n_qubits = math.ceil(d / 2)

        # Pad to even length if needed
        if d % 2 == 1:
            x_pad = np.append(x, 0.0)
        else:
            x_pad = x

        ry_angles = x_pad[0::2]  # even indices → Ry
        rz_angles = x_pad[1::2]  # odd  indices → Rz

        # Interleaved: [ry_0, rz_0, ry_1, rz_1, ...]
        parameters = np.empty(2 * n_qubits, dtype=float)
        parameters[0::2] = ry_angles
        parameters[1::2] = rz_angles

        return EncodedResult(
            parameters=parameters,
            metadata={
                "encoding": "tensor_product",
                "n_qubits": n_qubits,
                "depth": 2,
                "ry_angles": ry_angles.tolist(),
                "rz_angles": rz_angles.tolist(),
            },
        )
