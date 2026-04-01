r"""Pauli feature map encoding.

Mathematical formulation
------------------------
A generalization of the ZZ feature map that supports arbitrary Pauli
strings.  For each repetition:

$U_\Phi(x) = \prod_{P \in \mathcal{P}} U_P(x) \cdot H^{\otimes d}$

where $\mathcal{P}$ is a set of Pauli strings (e.g. ``['Z', 'ZZ']``).

For a single-qubit Pauli ``Z`` on qubit $i$:

$U_Z(x_i) = \text{Rz}(2x_i)$

For a two-qubit Pauli ``ZZ`` on qubits $i,j$:

$U_{ZZ}(x_i, x_j) = \text{CNOT}(i,j) \cdot \text{Rz}(2 x_i x_j) \cdot \text{CNOT}(i,j)$

Likewise ``XX`` and ``YY`` are implemented via basis changes
(Ry/Rx conjugation) around the same CNOT-Rz-CNOT building block.

Properties
----------
Qubits : n = d
Depth  : O(d² · reps) for pairwise Paulis
NISQ   : Medium to High depending on Pauli choice
Best for: Kernel methods requiring richer feature interactions.
"""

from __future__ import annotations

import numpy as np

from quprep.encode.base import BaseEncoder, EncodedResult

_VALID_SINGLE = {"X", "Y", "Z"}
_VALID_PAIR = {"XX", "YY", "ZZ", "XZ", "ZX", "XY", "YX", "YZ", "ZY"}
_VALID_PAULIS = _VALID_SINGLE | _VALID_PAIR


class PauliFeatureMapEncoder(BaseEncoder):
    """
    Pauli feature map encoder.

    Generalizes the ZZ feature map by allowing arbitrary single- and
    two-qubit Pauli strings as interaction terms.  The default
    ``paulis=['Z', 'ZZ']`` reproduces the ZZ feature map (with direct
    ``x_i`` parametrization rather than the ``π − xᵢ`` convention).

    Parameters
    ----------
    paulis : list of str
        Pauli strings to include.  Single-qubit: ``'X'``, ``'Y'``,
        ``'Z'``.  Two-qubit: ``'ZZ'``, ``'XX'``, ``'YY'``, ``'XZ'``,
        etc.  Default ``['Z', 'ZZ']``.
    reps : int
        Number of repetitions. Default 2.
    """

    def __init__(self, paulis: list[str] | None = None, reps: int = 2):
        if paulis is None:
            paulis = ["Z", "ZZ"]
        for p in paulis:
            if p not in _VALID_PAULIS:
                raise ValueError(
                    f"Unknown Pauli '{p}'. Valid options: {sorted(_VALID_PAULIS)}"
                )
        if reps < 1:
            raise ValueError(f"reps must be >= 1, got {reps}.")
        self.paulis = list(paulis)
        self.reps = reps

    @property
    def n_qubits(self):
        return None  # data-dependent

    @property
    def depth(self):
        has_pair = any(len(p) == 2 for p in self.paulis)
        return "O(d² · reps)" if has_pair else "O(d · reps)"

    def encode(self, x: np.ndarray) -> EncodedResult:
        r"""
        Encode a 1-D feature vector using the Pauli feature map.

        Parameters
        ----------
        x : np.ndarray, shape (d,)
            Normalized feature vector in $[-\pi, \pi]$.
            Use ``Scaler('minmax_pm_pi')``.

        Returns
        -------
        EncodedResult
            ``parameters`` = ``x``.
            ``metadata`` includes ``encoding``, ``n_qubits``, ``paulis``,
            ``reps``, ``single_terms``, ``pair_terms``.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or len(x) == 0:
            raise ValueError("PauliFeatureMapEncoder.encode() expects a non-empty 1-D array.")

        d = len(x)

        # Build single-qubit terms: {pauli_str: [angle_0, ..., angle_{d-1}]}
        single_terms: dict[str, list[float]] = {}
        for p in self.paulis:
            if len(p) == 1:
                single_terms[p] = (2.0 * x).tolist()

        # Build pairwise terms: {pauli_str: [(i, j, angle), ...]}
        pair_terms: dict[str, list[tuple[int, int, float]]] = {}
        for p in self.paulis:
            if len(p) == 2:
                entries = [
                    (i, j, float(2.0 * x[i] * x[j]))
                    for i in range(d)
                    for j in range(i + 1, d)
                ]
                pair_terms[p] = entries

        return EncodedResult(
            parameters=x.copy(),
            metadata={
                "encoding": "pauli_feature_map",
                "n_qubits": d,
                "paulis": self.paulis,
                "reps": self.reps,
                "depth": d * d * self.reps if pair_terms else d * self.reps,
                "single_terms": single_terms,
                "pair_terms": pair_terms,
            },
        )
