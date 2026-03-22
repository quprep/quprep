r"""Amplitude encoding — embeds data as quantum state amplitudes.

Mathematical formulation
------------------------
Given L2-normalized $x \in \mathbb{R}^d$ with $\|x\|_2 = 1$:

$|\psi(x)\rangle = \sum_{i=0}^{d-1} x_i |i\rangle$

Requires $d = 2^n$. If d is not a power of two, pad with zeros.

Properties
----------
Qubits : $n = \lceil \log_2(d) \rceil$
Depth  : $O(2^n)$ — exponential state preparation circuit.
NISQ   : Poor — deep circuit, not suitable for current hardware.
Best for: Qubit-limited scenarios where expressivity matters more than depth.

Reference: Mottonen et al., "Decomposition of arbitrary quantum gates"
"""

from __future__ import annotations

import numpy as np

from quprep.encode.base import BaseEncoder, EncodedResult


class AmplitudeEncoder(BaseEncoder):
    r"""
    Amplitude encoding.

    x must be L2-normalized ($\|x\|_2 = 1$). Use quprep.normalize.Scaler('l2').

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
        r"""
        Encode x as amplitude vector.

        Parameters
        ----------
        x : np.ndarray, shape (d,)
            L2-normalized feature vector ($\|x\|_2 = 1$). Use ``Scaler('l2')``.
            If d is not a power of two and ``pad=True``, zero-pads and re-normalizes.

        Returns
        -------
        EncodedResult
            ``parameters`` = amplitude vector (length padded to next power of two).
            ``metadata`` includes ``encoding``, ``n_qubits``, ``padded``, ``original_dim``.
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
