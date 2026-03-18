"""Normalization strategies with auto-selection per encoding type.

Encoding–normalization mapping
-------------------------------
Amplitude  → L2-normalize (‖x‖₂ = 1). Amplitudes must form a valid quantum state.
Angle Ry   → scale to [0, π]. Maps to rotation angles on the Bloch sphere.
Angle Rx   → scale to [−π, π].
Basis      → binarize. Qubits are |0⟩ or |1⟩ only.
IQP        → scale to [−π, π] and compute feature products xᵢ·xⱼ.
QUBO/Ising → binary {0,1} or integer {−1,+1} discretization.

Users can override by passing a normalizer explicitly to the Pipeline.
"""

from __future__ import annotations

import numpy as np


ENCODING_NORMALIZER_MAP: dict[str, str] = {
    "amplitude": "l2",
    "angle_ry": "minmax_pi",
    "angle_rx": "minmax_pm_pi",
    "angle_rz": "minmax_pm_pi",
    "basis": "binary",
    "iqp": "minmax_pm_pi",
    "qubo": "binary",
    "ising": "pm_one",
}


def auto_normalizer(encoding: str) -> "Scaler":
    """Return the correct Scaler for a given encoding name."""
    key = ENCODING_NORMALIZER_MAP.get(encoding)
    if key is None:
        raise ValueError(f"Unknown encoding '{encoding}'. Known: {list(ENCODING_NORMALIZER_MAP)}")
    return Scaler(strategy=key)


class Scaler:
    """
    Apply a normalization strategy to a Dataset.

    Parameters
    ----------
    strategy : str
        'l2'           — unit L2 norm per sample (amplitude encoding).
        'minmax_pi'    — scale to [0, π] (angle Ry).
        'minmax_pm_pi' — scale to [−π, π] (angle Rx/Rz, IQP).
        'zscore'       — zero mean, unit variance.
        'minmax'       — scale to [0, 1].
        'binary'       — threshold to {0, 1}.
        'pm_one'       — map to {−1, +1} (Ising).
    """

    def __init__(self, strategy: str = "minmax"):
        self.strategy = strategy

    def fit_transform(self, dataset):
        """Normalize and return Dataset."""
        raise NotImplementedError("Scaler.fit_transform() — coming in v0.1.0")
