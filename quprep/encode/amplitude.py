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

        Validates ‖x‖₂ = 1 (within numerical tolerance). If d is not a power
        of two, pads with zeros and re-normalizes (when pad=True).
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValueError(f"Expected 1D input, got shape {x.shape}")
        if len(x) == 0:
            raise ValueError("Input vector must not be empty")

        norm = np.linalg.norm(x)
        if not np.isclose(norm, 1.0, atol=1e-6):
            raise ValueError(
                f"AmplitudeEncoder requires L2-normalized input (‖x‖₂ = 1), "
                f"got ‖x‖₂ = {norm:.6f}. Use Scaler('l2') first."
            )

        d = len(x)
        next_pow2 = 1 << (d - 1).bit_length() if d > 1 else 1
        if d != next_pow2:
            if not self.pad:
                raise ValueError(
                    f"d={d} is not a power of two. Set pad=True to zero-pad, "
                    f"or reduce dimensions to d={next_pow2}."
                )
            padded = np.zeros(next_pow2, dtype=float)
            padded[:d] = x
            padded /= np.linalg.norm(padded)  # re-normalize after padding
        else:
            padded = x.copy()

        n_qubits = int(np.log2(len(padded)))
        return EncodedResult(
            parameters=padded,
            metadata={
                "encoding": "amplitude",
                "n_qubits": n_qubits,
                "depth": "O(2^n)",
                "padded": d != next_pow2,
                "original_dim": d,
            },
        )
