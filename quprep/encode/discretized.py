r"""Discretized encoding — continuous features as fixed-point binary basis states.

Mathematical formulation
------------------------
Each continuous feature $x_i \in [\text{min}, \text{max}]$ is quantized to
``bits`` binary digits using standard unsigned fixed-point representation.
The integer value is:

$v_i = \text{round}\!\left(\frac{x_i - x_{\min}}{x_{\max} - x_{\min}} \cdot (2^b - 1)\right)$

and the binary expansion (MSB-first) $b_{i,0}, \ldots, b_{i,b-1}$ satisfies:

$v_i = \sum_{k=0}^{b-1} b_{i,k} \cdot 2^{b-1-k}$

The full circuit state is a computational basis state:

$|\psi(x)\rangle = \bigotimes_{i=0}^{d-1} |b_{i,0}\, b_{i,1}\, \cdots\, b_{i,b-1}\rangle$

set by applying X gates on qubits where $b_{i,k} = 1$.

QUBO compatibility
------------------
The output binary vector maps directly to QUBO decision variables.
``metadata['qubo_variables']`` provides the qubit-index slice for each feature,
so you can pass the vector directly to :func:`quprep.qubo.to_qubo`:

    enc = DiscretizedEncoder(bits=4)
    result = enc.encode(x)
    binary_vars = result.parameters  # shape (d * bits,)

Properties
----------
Qubits : d × bits
Depth  : 1 (only X gates)
NISQ   : Excellent — single-layer, hardware-native.
Best for: Connecting encode pipeline to QUBO/Ising optimization; continuous
          relaxations of combinatorial problems; QAOA warm-starting.
"""

from __future__ import annotations

import numpy as np

from quprep.encode.base import BaseEncoder, EncodedResult


class DiscretizedEncoder(BaseEncoder):
    """
    Discretized encoding — maps continuous features to binary basis states.

    Each feature is quantized into ``bits`` binary digits using unsigned
    fixed-point representation. The resulting binary vector is QUBO-ready
    and can be passed directly to :func:`quprep.qubo.to_qubo`.

    Parameters
    ----------
    bits : int
        Bits per feature. Default 4.
        Precision = (max_val − min_val) / (2^bits − 1).
    min_val : float
        Lower bound of the expected feature range. Default 0.0.
    max_val : float
        Upper bound of the expected feature range. Default 1.0.
    """

    def __init__(self, bits: int = 4, min_val: float = 0.0, max_val: float = 1.0):
        if bits < 1:
            raise ValueError(f"bits must be >= 1, got {bits}")
        if min_val >= max_val:
            raise ValueError(
                f"min_val must be strictly less than max_val, "
                f"got min_val={min_val}, max_val={max_val}"
            )
        self.bits = bits
        self.min_val = float(min_val)
        self.max_val = float(max_val)

    @property
    def n_qubits(self) -> None:
        return None  # data-dependent: d * bits

    @property
    def depth(self) -> int:
        return 1

    def encode(self, x: np.ndarray) -> EncodedResult:
        """
        Quantize each feature to ``bits`` binary digits and encode as a basis state.

        Parameters
        ----------
        x : np.ndarray, shape (d,)
            Feature vector. Values outside ``[min_val, max_val]`` are clipped.

        Returns
        -------
        EncodedResult
            ``parameters``: binary float array of shape ``(d * bits,)``, MSB-first
            per feature. ``metadata`` includes ``encoding``, ``n_qubits``, ``bits``,
            ``min_val``, ``max_val``, ``precision``, and ``qubo_variables`` (a dict
            mapping each feature index to its list of qubit indices).
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or len(x) == 0:
            raise ValueError("DiscretizedEncoder.encode() expects a non-empty 1-D array.")

        d = len(x)
        n_qubits = d * self.bits
        levels = (1 << self.bits) - 1  # 2^bits − 1

        x_clipped = np.clip(x, self.min_val, self.max_val)
        normalized = (x_clipped - self.min_val) / (self.max_val - self.min_val)
        integers = np.round(normalized * levels).astype(int)

        bits_array = np.zeros(n_qubits, dtype=float)
        for i, val in enumerate(integers):
            for k in range(self.bits):
                bits_array[i * self.bits + k] = float((val >> (self.bits - 1 - k)) & 1)

        qubo_variables = {
            i: list(range(i * self.bits, (i + 1) * self.bits)) for i in range(d)
        }

        return EncodedResult(
            parameters=bits_array,
            metadata={
                "encoding": "discretized",
                "n_qubits": n_qubits,
                "bits": self.bits,
                "min_val": self.min_val,
                "max_val": self.max_val,
                "precision": (self.max_val - self.min_val) / levels,
                "depth": 1,
                "qubo_variables": qubo_variables,
            },
        )

    def decode(self, bits_array: np.ndarray) -> np.ndarray:
        """
        Reconstruct continuous feature values from a binary parameter array.

        Parameters
        ----------
        bits_array : np.ndarray, shape (d * bits,)
            Binary array as returned by ``encode().parameters``.

        Returns
        -------
        np.ndarray, shape (d,)
            Reconstructed feature values in ``[min_val, max_val]``.
        """
        bits_array = np.asarray(bits_array, dtype=float)
        n = len(bits_array)
        if n % self.bits != 0:
            raise ValueError(
                f"bits_array length {n} is not divisible by bits={self.bits}"
            )
        d = n // self.bits
        levels = (1 << self.bits) - 1
        result = np.zeros(d)
        for i in range(d):
            chunk = bits_array[i * self.bits: (i + 1) * self.bits]
            val = int(
                sum(int(b) * (1 << (self.bits - 1 - k)) for k, b in enumerate(chunk))
            )
            result[i] = self.min_val + (self.max_val - self.min_val) * val / levels
        return result
