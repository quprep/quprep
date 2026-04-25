"""Expressibility and entanglement capability metrics (Sim et al. 2019).

References
----------
Sim S. et al. "Expressibility and Entangling Capability of Parameterized
Quantum Circuits for Hybrid Quantum-Classical Algorithms."
*Advanced Quantum Technologies* 2(12), 2019.
https://doi.org/10.1002/qute.201900070
"""

from __future__ import annotations

import numpy as np

from quprep.metrics._simulate import statevector_from_encoded

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fidelity(sv1: np.ndarray, sv2: np.ndarray) -> float:
    """Compute |⟨ψ₁|ψ₂⟩|²."""
    return float(abs(np.dot(sv1.conj(), sv2)) ** 2)


def _haar_pdf(F: np.ndarray, n_qubits: int) -> np.ndarray:
    """Haar fidelity distribution: (2ⁿ − 1)(1 − F)^(2ⁿ − 2)."""
    N = 1 << n_qubits
    return (N - 1) * np.power(np.clip(1.0 - F, 0.0, 1.0), N - 2, dtype=float)


def _meyer_wallach(sv: np.ndarray, n_qubits: int) -> float:
    """Meyer-Wallach entanglement measure Q ∈ [0, 1].

    Q = (2/n) Σ_k (1 − Tr(ρ_k²))  where ρ_k is the reduced DM of qubit k.
    """
    if n_qubits == 1:
        return 0.0
    state = sv.reshape([2] * n_qubits)
    total = 0.0
    for k in range(n_qubits):
        moved = np.moveaxis(state, k, 0)      # qubit k → axis 0
        sub = moved.reshape(2, -1)            # (2, 2^(n-1))
        rho = sub @ sub.conj().T              # 2×2 reduced density matrix
        total += 1.0 - float(np.real(np.trace(rho @ rho)))
    return (2.0 / n_qubits) * total


def _sample_rows(dataset, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n rows from dataset (with replacement when needed)."""
    data = dataset.data
    m = len(data)
    idx = rng.choice(m, size=n, replace=(m < n))
    return data[idx]


def _encode_statevectors(
    encoder, rows: np.ndarray
) -> tuple[list[np.ndarray], int] | tuple[None, None]:
    """Encode each row and return (statevectors, n_qubits), or (None, None) on failure."""
    svs: list[np.ndarray] = []
    n_qubits: int | None = None
    for row in rows:
        try:
            enc = encoder.encode(row)
        except Exception:
            continue
        sv = statevector_from_encoded(enc)
        if sv is None:
            return None, None
        if n_qubits is None:
            n_qubits = enc.metadata.get("n_qubits", int(round(np.log2(len(sv)))))
        svs.append(sv)
    if not svs or n_qubits is None:
        return None, None
    return svs, n_qubits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def expressibility(
    encoder,
    dataset,
    *,
    n_samples: int = 500,
    n_bins: int = 75,
    seed: int | None = None,
) -> float | None:
    """
    Estimate the expressibility of an encoding as KL divergence from Haar.

    A lower value indicates a more expressive circuit (closer to the
    uniformly-random Haar distribution over the Hilbert space).

    Parameters
    ----------
    encoder : encoder instance
        A fitted QuPrep encoder (e.g. ``AngleEncoder()``, ``IQPEncoder()``).
        ``RandomFourierEncoder`` must be fitted before calling this function.
    dataset : Dataset
        Source data — rows are sampled to build the fidelity distribution.
    n_samples : int
        Number of data rows to sample for fidelity estimation. Default 500.
    n_bins : int
        Number of histogram bins for the KL divergence estimate. Default 75.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    float or None
        KL divergence ≥ 0.  ``None`` if the encoding is unsupported or
        ``n_qubits > metrics.MAX_QUBITS``.

    References
    ----------
    Sim et al. (2019) https://doi.org/10.1002/qute.201900070
    """
    rng = np.random.default_rng(seed)
    rows = _sample_rows(dataset, n_samples, rng)
    svs, n_qubits = _encode_statevectors(encoder, rows)
    if svs is None or n_qubits is None:
        return None

    m = len(svs)
    max_pairs = 2_000
    total_pairs = m * (m - 1) // 2

    if total_pairs <= max_pairs:
        fidelities = [_fidelity(svs[i], svs[j]) for i in range(m) for j in range(i + 1, m)]
    else:
        fidelities = []
        for _ in range(max_pairs):
            i, j = rng.choice(m, size=2, replace=False)
            fidelities.append(_fidelity(svs[i], svs[j]))

    F = np.array(fidelities, dtype=float)
    counts, edges = np.histogram(F, bins=n_bins, range=(0.0, 1.0), density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    bw = edges[1] - edges[0]

    P = counts * bw + 1e-10
    Q = _haar_pdf(centers, n_qubits) * bw + 1e-10
    P /= P.sum()
    Q /= Q.sum()

    return float(max(np.sum(P * np.log(P / Q)), 0.0))


def entanglement_capability(
    encoder,
    dataset,
    *,
    n_samples: int = 200,
    seed: int | None = None,
) -> float | None:
    """
    Estimate the entanglement capability of an encoding.

    Returns the average Meyer-Wallach measure over randomly sampled data
    points.  Ranges from 0 (product state, e.g. plain angle encoding) to 1
    (maximally entangled).

    Parameters
    ----------
    encoder : encoder instance
    dataset : Dataset
    n_samples : int
        Default 200.
    seed : int, optional

    Returns
    -------
    float or None
        Average MW measure ∈ [0, 1].  ``None`` if unsupported or too large.

    References
    ----------
    Sim et al. (2019) https://doi.org/10.1002/qute.201900070
    """
    rng = np.random.default_rng(seed)
    rows = _sample_rows(dataset, n_samples, rng)
    svs, n_qubits = _encode_statevectors(encoder, rows)
    if svs is None or n_qubits is None:
        return None
    values = [_meyer_wallach(sv, n_qubits) for sv in svs]
    return float(np.mean(values))
