"""Quantum kernel alignment and composite encoder quality metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quprep.metrics._simulate import statevector_from_encoded

# ---------------------------------------------------------------------------
# Kernel alignment
# ---------------------------------------------------------------------------

def kernel_alignment(
    encoder,
    dataset,
    *,
    max_samples: int = 300,
    seed: int | None = None,
) -> float | None:
    """
    Compute the normalised kernel alignment between the quantum kernel and labels.

    Measures how well the encoding separates classes by comparing the quantum
    kernel matrix K (where K[i,j] = |⟨ψ(xᵢ)|ψ(xⱼ)⟩|²) to the ideal label
    kernel K_y (where K_y[i,j] = yᵢ·yⱼ).

    The alignment is:

    .. math::

        A(K, K_y) = \\frac{\\langle K, K_y \\rangle_F}{\\|K\\|_F \\|K_y\\|_F}

    Higher values indicate the encoding separates classes better.

    Parameters
    ----------
    encoder : encoder instance
        A fitted QuPrep encoder.
    dataset : Dataset
        Must have ``dataset.labels`` populated.
    max_samples : int
        Subsample the dataset to at most this many points for efficiency.
        Default 300.
    seed : int, optional

    Returns
    -------
    float or None
        Alignment score ∈ [−1, 1].  ``None`` if labels are missing,
        encoding unsupported, or ``n_qubits > metrics.MAX_QUBITS``.
    """
    if dataset.labels is None:
        return None

    rng = np.random.default_rng(seed)
    X = dataset.data
    y = np.asarray(dataset.labels, dtype=float)
    if y.ndim > 1:
        y = y[:, 0]  # use first label column for multi-output

    n = len(X)
    if n < 4:
        return None

    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        X, y = X[idx], y[idx]
        n = max_samples

    # Encode all rows
    svs: list[np.ndarray] = []
    for row in X:
        try:
            enc = encoder.encode(row)
        except Exception:
            return None
        sv = statevector_from_encoded(enc)
        if sv is None:
            return None
        svs.append(sv)

    S = np.array(svs)                      # (n, 2^d) complex
    K = np.abs(S @ S.conj().T) ** 2        # (n, n) real quantum kernel
    K_y = np.outer(y, y)                   # (n, n) label kernel

    inner = float(np.sum(K * K_y))
    norm_k = float(np.sqrt(np.sum(K ** 2)))
    norm_y = float(np.sqrt(np.sum(K_y ** 2)))

    if norm_k < 1e-12 or norm_y < 1e-12:
        return None
    return inner / (norm_k * norm_y)


# ---------------------------------------------------------------------------
# Composite quality dataclass + score_encoding
# ---------------------------------------------------------------------------

@dataclass
class EncoderMetrics:
    """
    Data-driven quality metrics for a parameterized encoding on a dataset.

    Attributes
    ----------
    encoding : str
        Encoder name (e.g. ``'iqp'``).
    expressibility : float or None
        KL divergence from the Haar distribution.  **Lower = more expressive.**
        ``None`` if the circuit is too large to simulate.
    entanglement_capability : float or None
        Average Meyer-Wallach entanglement measure ∈ [0, 1].
        **Higher = more entangled.**  0 for product-state encodings.
        ``None`` if unsupported.
    kernel_alignment : float or None
        Normalised Frobenius alignment of the quantum kernel with class labels,
        ∈ [−1, 1].  **Higher = better class separation.**
        ``None`` if labels are not available.
    n_qubits : int
        Qubit count used for simulation.
    """

    encoding: str
    expressibility: float | None
    entanglement_capability: float | None
    kernel_alignment: float | None
    n_qubits: int

    def __str__(self) -> str:
        def _fmt(v: float | None) -> str:
            return f"{v:.4f}" if v is not None else "n/a"

        return (
            f"EncoderMetrics({self.encoding})\n"
            f"  expressibility         : {_fmt(self.expressibility)}"
            + (" (lower = better)" if self.expressibility is not None else "") + "\n"
            f"  entanglement_capability: {_fmt(self.entanglement_capability)}"
            + (" (higher = better)" if self.entanglement_capability is not None else "")
            + "\n"
            f"  kernel_alignment       : {_fmt(self.kernel_alignment)}"
            + (" (higher = better)" if self.kernel_alignment is not None else "") + "\n"
            f"  n_qubits               : {self.n_qubits}"
        )


def score_encoding(
    encoder,
    dataset,
    *,
    n_samples: int = 200,
    seed: int | None = None,
) -> EncoderMetrics:
    """
    Compute all data-driven quality metrics for one encoder on a dataset.

    Encoders that require fitting (e.g. ``RandomFourierEncoder``) are
    automatically fitted on the dataset before metric computation.

    Parameters
    ----------
    encoder : encoder instance
    dataset : Dataset
    n_samples : int
        Samples used for expressibility and entanglement. Default 200.
    seed : int, optional

    Returns
    -------
    EncoderMetrics
    """
    from quprep.metrics.expressibility import entanglement_capability, expressibility

    # Fit encoder if it exposes a fit() method and hasn't been fitted yet.
    if hasattr(encoder, "fit") and not getattr(encoder, "_fitted", True):
        try:
            encoder.fit(dataset)
        except Exception:
            pass
    # RandomFourierEncoder stores fitted state differently — try fit always.
    elif hasattr(encoder, "_W") is False and hasattr(encoder, "fit"):
        try:
            encoder.fit(dataset)
        except Exception:
            pass

    enc_name: str = getattr(encoder, "encoding", type(encoder).__name__)

    # Determine n_qubits from one encode call
    n_qubits = 0
    try:
        enc_result = encoder.encode(dataset.data[0])
        n_qubits = enc_result.metadata.get("n_qubits", 0)
    except Exception:
        pass

    exp = expressibility(encoder, dataset, n_samples=n_samples, seed=seed)
    ent = entanglement_capability(
        encoder, dataset, n_samples=n_samples // 2, seed=seed
    )
    ka = kernel_alignment(encoder, dataset, seed=seed)

    return EncoderMetrics(
        encoding=enc_name,
        expressibility=exp,
        entanglement_capability=ent,
        kernel_alignment=ka,
        n_qubits=n_qubits,
    )
