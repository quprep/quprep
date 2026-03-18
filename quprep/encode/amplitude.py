"""Amplitude encoding — embeds data as quantum state amplitudes.

Mathematical formulation
------------------------
Given L2-normalized x ∈ ℝ^d with ‖x‖₂ = 1:

    |ψ(x)⟩ = Σ_{i=0}^{d-1} x_i |i⟩

Requires d = 2^n. If d is not a power of two, pad with zeros.

Properties
----------
Qubits : n = ⌈log₂(d)⌉
Depth  : O(2^n) — exponential state preparation circuit.
NISQ   : Poor — deep circuit, not suitable for current hardware.
Best for: Qubit-limited scenarios where expressivity matters more than depth.

Reference: Mottonen et al., "Decomposition of arbitrary quantum gates"
"""

from __future__ import annotations

import numpy as np

from quprep.encode.base import BaseEncoder, EncodedResult


class AmplitudeEncoder(BaseEncoder):
    """
    Amplitude encoding.

    x must be L2-normalized (‖x‖₂ = 1). Use quprep.normalize.Scaler('l2').

    Parameters
    ----------
    pad : bool
        If True, zero-pad x to the next power of two when d is not 2^n.
        If False, raise ValueError.
    """

    def __init__(self, pad: bool = True):
        self.pad = pad

    @property
    def n_qubits(self):
        return None  # data-dependent: n_qubits = ceil(log2(d))

    @property
    def depth(self):
        return "O(2^n)"

    def encode(self, x: np.ndarray) -> EncodedResult:
        """
        Encode x as amplitude vector.

        Validates ‖x‖₂ = 1 (within numerical tolerance).
        """
        raise NotImplementedError("AmplitudeEncoder.encode() — coming in v0.1.0")
