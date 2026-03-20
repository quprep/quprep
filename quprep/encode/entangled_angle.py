"""Entangled angle encoding — rotation gates interleaved with entangling CNOT layers.

Mathematical formulation
------------------------
Applies ``layers`` repetitions of:

    1. Single-qubit rotation layer:  R_G(x_i) on each qubit i
    2. Entangling layer:             CNOT gates according to ``entanglement``

where R_G ∈ {Ry, Rx, Rz}.

Entanglement patterns
---------------------
- ``linear``   : CNOT(0,1), CNOT(1,2), ..., CNOT(d-2, d-1)  — chain topology
- ``circular`` : linear + CNOT(d-1, 0)                       — ring topology
- ``full``     : CNOT(i, j) for all i < j                    — all-to-all

Properties
----------
Qubits : n = d
Depth  : O(d · layers)  [linear/circular]  or  O(d² · layers)  [full]
NISQ   : Good for linear/circular (shallow per layer). Full entanglement
         may be deep on near-term hardware.
Best for: Classification and QML tasks that benefit from feature correlations.
"""

from __future__ import annotations

import numpy as np

from quprep.encode.base import BaseEncoder, EncodedResult

_ENTANGLEMENT_PATTERNS = ("linear", "circular", "full")


class EntangledAngleEncoder(BaseEncoder):
    """
    Angle encoding with entangling CNOT layers between rotation rounds.

    Each layer applies single-qubit rotations on all qubits followed by
    a structured CNOT entangling block. Features are re-uploaded in every
    layer (data re-uploading style).

    Parameters
    ----------
    rotation : str
        Rotation gate: ``'ry'`` (default), ``'rx'``, or ``'rz'``.
    layers : int
        Number of rotation+entanglement repetitions. Default 1.
    entanglement : str
        CNOT topology: ``'linear'`` (default), ``'circular'``, or ``'full'``.
    """

    def __init__(
        self,
        rotation: str = "ry",
        layers: int = 1,
        entanglement: str = "linear",
    ):
        if rotation not in ("ry", "rx", "rz"):
            raise ValueError(f"rotation must be 'ry', 'rx', or 'rz', got '{rotation}'")
        if layers < 1:
            raise ValueError(f"layers must be >= 1, got {layers}")
        if entanglement not in _ENTANGLEMENT_PATTERNS:
            raise ValueError(
                f"entanglement must be one of {_ENTANGLEMENT_PATTERNS}, got '{entanglement}'"
            )
        self.rotation = rotation
        self.layers = layers
        self.entanglement = entanglement

    @property
    def n_qubits(self):
        return None  # data-dependent

    @property
    def depth(self):
        return None  # data-dependent

    def encode(self, x: np.ndarray) -> EncodedResult:
        """
        Encode feature vector x as entangled rotation angles.

        Parameters
        ----------
        x : np.ndarray, shape (d,)
            Normalized feature vector. Use ``auto_normalizer`` for correct
            scaling (same as AngleEncoder: ``minmax_pi`` for Ry, ``minmax_pm_pi``
            for Rx/Rz).

        Returns
        -------
        EncodedResult
            ``parameters`` = x (stored once; exporter repeats per layer).
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValueError(f"Expected 1D input, got shape {x.shape}")
        if len(x) == 0:
            raise ValueError("Input vector must not be empty")

        d = len(x)
        # Compute CNOT pairs for metadata (so exporters don't need to recompute)
        cnot_pairs = _cnot_pairs(d, self.entanglement)

        # Depth: d rotation gates + len(cnot_pairs) CNOTs per layer
        circuit_depth = (d + len(cnot_pairs)) * self.layers

        return EncodedResult(
            parameters=x.copy(),
            metadata={
                "encoding": "entangled_angle",
                "rotation": self.rotation,
                "n_qubits": d,
                "layers": self.layers,
                "entanglement": self.entanglement,
                "cnot_pairs": cnot_pairs,
                "depth": circuit_depth,
            },
        )


def _cnot_pairs(d: int, entanglement: str) -> list[tuple[int, int]]:
    """Return list of (control, target) CNOT pairs for a given pattern."""
    if d < 2:
        return []
    if entanglement == "linear":
        return [(i, i + 1) for i in range(d - 1)]
    if entanglement == "circular":
        return [(i, i + 1) for i in range(d - 1)] + [(d - 1, 0)]
    # full
    return [(i, j) for i in range(d) for j in range(i + 1, d)]
