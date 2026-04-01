r"""QAOA-inspired problem encoder.

Mathematical formulation
------------------------
Feature vector $x \in [-\pi, \pi]^d$ is treated as a one-layer QAOA Hamiltonian,
mapping data into the variational manifold of a parameterised QAOA circuit.

Circuit structure ($p$ layers):

1. **Initialization**: $H^{\otimes d}|0\rangle^d \rightarrow |{+}\rangle^d$
2. For each layer $l$:

   a. **Cost unitary** $U_C(\gamma, x)$:

      - Single-qubit: $\text{RZ}(2\gamma x_i)$ on qubit $i$
      - Pairwise (linear): CNOT-RZ($2\gamma x_i x_{i+1}$)-CNOT on adjacent pairs
      - Pairwise (full): all $\binom{d}{2}$ pairs

   b. **Mixer unitary** $U_B(\beta)$: $\text{RX}(2\beta)$ on each qubit $i$

The feature-dependent state $|\psi(x)\rangle = U_B(\beta) U_C(\gamma, x) |{+}\rangle^d$
defines a data-dependent kernel implicitly, similar in spirit to IQP but with
the shallow QAOA structure rather than diagonal unitaries.

Properties
----------
Qubits : n = d
Depth  : O(p) with linear connectivity; O(d · p) with full connectivity.
NISQ   : Yes (linear, p=1) — depth ~6 for d features, 2(d−1) two-qubit gates.
Best for: QAOA warm-starting, problem-inspired feature maps, NISQ kernel methods.

References
----------
Farhi, E., Goldstone, J., Gutmann, S. (2014). A Quantum Approximate
    Optimization Algorithm. arXiv:1411.4028.
Cerezo, M. et al. (2021). Variational quantum algorithms. Nature Reviews
    Physics 3, 625–644.
"""

from __future__ import annotations

import numpy as np

from quprep.encode.base import BaseEncoder, EncodedResult


class QAOAProblemEncoder(BaseEncoder):
    r"""
    QAOA-inspired problem encoder.

    Encodes a feature vector as a one-layer (or multi-layer) QAOA circuit
    where the cost Hamiltonian is constructed from the feature values:

    - Local fields: $h_i = \gamma \cdot x_i$
    - Couplings:    $J_{ij} = \gamma \cdot x_i x_j$ (linear or full connectivity)

    Parameters
    ----------
    p : int
        Number of QAOA layers (circuit repetitions). Default 1.
    gamma : float
        Scaling factor for the cost unitary angles. Default ``π/4``.
    beta : float
        Mixing angle for the RX mixer. Default ``π/8``.
    connectivity : {"linear", "full"}
        ``"linear"`` (default) — only adjacent qubit pairs, NISQ-safe.
        ``"full"`` — all :math:`\binom{d}{2}` pairs, $O(d^2)$ depth.
    """

    def __init__(
        self,
        p: int = 1,
        gamma: float = np.pi / 4,
        beta: float = np.pi / 8,
        connectivity: str = "linear",
    ):
        if p < 1:
            raise ValueError(f"p must be >= 1, got {p}.")
        if connectivity not in ("linear", "full"):
            raise ValueError(f"connectivity must be 'linear' or 'full', got '{connectivity}'.")
        self.p = p
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.connectivity = connectivity

    @property
    def n_qubits(self):
        return None  # data-dependent: n_qubits = n_features

    @property
    def depth(self):
        if self.connectivity == "linear":
            return "O(p)"
        return "O(d · p)"

    def encode(self, x: np.ndarray) -> EncodedResult:
        r"""
        Encode a 1-D feature vector as a QAOA problem circuit.

        Parameters
        ----------
        x : np.ndarray, shape (d,)
            Normalized feature vector in $[-\pi, \pi]$.
            Use ``Scaler('minmax_pm_pi')`` for correct scaling.

        Returns
        -------
        EncodedResult
            ``parameters`` = $[\gamma x_0, \ldots, \gamma x_{d-1},
            \gamma x_0 x_1, \ldots]$ (local angles then coupling angles).
            ``metadata`` includes ``encoding``, ``n_qubits``, ``p``,
            ``gamma``, ``beta``, ``connectivity``, ``depth``.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or len(x) == 0:
            raise ValueError("QAOAProblemEncoder.encode() expects a non-empty 1-D array.")

        d = len(x)

        # Local field angles: γ * x_i (one per qubit)
        local_angles = self.gamma * x

        # Coupling angles: γ * x_i * x_j for connected pairs
        if self.connectivity == "linear":
            pairs = [(i, i + 1) for i in range(d - 1)]
        else:  # full
            pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]

        coupling_angles = np.array(
            [self.gamma * x[i] * x[j] for i, j in pairs],
            dtype=float,
        )

        parameters = np.concatenate([local_angles, coupling_angles])

        # Depth calculation
        if self.connectivity == "linear":
            # H(1) + per layer: RZ(1) + CNOT-RZ-CNOT(3) + RX(1) = 5 + H init
            depth = 1 + 5 * self.p
        else:
            # Full connectivity serialized: each pair ~3 gates deep
            depth = 1 + (1 + 3 * len(pairs) + 1) * self.p

        return EncodedResult(
            parameters=parameters,
            metadata={
                "encoding": "qaoa_problem",
                "n_qubits": d,
                "p": self.p,
                "gamma": self.gamma,
                "beta": self.beta,
                "connectivity": self.connectivity,
                "depth": depth,
                "n_pairs": len(pairs),
                "pairs": pairs,
                "local_angles": local_angles.tolist(),
                "coupling_angles": coupling_angles.tolist(),
            },
        )
