"""Hamiltonian encoding — time-evolution encoding for physics simulations.

Mathematical formulation
------------------------
Given x ∈ ℝ^d encoding Hamiltonian parameters and evolution time T:

    |ψ(x)⟩ = e^{-i H(x) T} |0⟩^n

where H(x) = Σ_i x_i Z_i (single-qubit Pauli Z Hamiltonian).

Trotterized as S Rz(2·x_i·T/S) repetitions per qubit, where S = trotter_steps.

Properties
----------
Qubits : n = d
Depth  : O(d · trotter_steps)
NISQ   : Poor — requires many repetitions for high-fidelity simulation.
Best for: Physics simulation, VQE.
"""

from __future__ import annotations

import numpy as np

from quprep.encode.base import BaseEncoder, EncodedResult


class HamiltonianEncoder(BaseEncoder):
    """
    Hamiltonian / time-evolution encoding.

    Parameters
    ----------
    evolution_time : float
        Evolution time T.
    trotter_steps : int
        Number of Trotter steps S. More steps → higher fidelity, greater depth.
    """

    def __init__(self, evolution_time: float = 1.0, trotter_steps: int = 4):
        if trotter_steps < 1:
            raise ValueError(f"trotter_steps must be >= 1, got {trotter_steps}.")
        if evolution_time <= 0:
            raise ValueError(f"evolution_time must be > 0, got {evolution_time}.")
        self.evolution_time = evolution_time
        self.trotter_steps = trotter_steps

    @property
    def n_qubits(self):
        return None  # data-dependent

    @property
    def depth(self):
        return "O(d · trotter_steps)"

    def encode(self, x: np.ndarray) -> EncodedResult:
        """
        Encode a 1-D feature vector using Trotterized Hamiltonian evolution.

        Parameters store per-step Rz angles: 2·x_i·T/S for each feature.
        The exporter applies these `trotter_steps` times per qubit.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or len(x) == 0:
            raise ValueError("HamiltonianEncoder.encode() expects a non-empty 1-D array.")

        d = len(x)
        # Per-step rotation angle: 2 * x_i * T / S
        angles = 2.0 * x * self.evolution_time / self.trotter_steps

        return EncodedResult(
            parameters=angles,
            metadata={
                "encoding": "hamiltonian",
                "n_qubits": d,
                "trotter_steps": self.trotter_steps,
                "evolution_time": self.evolution_time,
                "depth": d * self.trotter_steps,
            },
        )
