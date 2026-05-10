"""Encoding sensitivity analysis — per-feature influence on the quantum state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SensitivityResult:
    """
    Per-feature sensitivity scores for an encoding.

    Attributes
    ----------
    feature_names : list[str]
    scores : np.ndarray
        Sensitivity score per feature: mean state infidelity
        ``(1 − |⟨ψ|ψ'⟩|²)`` when that feature is perturbed by *epsilon*.
        Higher = more influential on the circuit output.
    epsilon : float
        Perturbation magnitude used.
    n_samples : int
        Number of dataset samples evaluated.
    """

    feature_names: list[str]
    scores: np.ndarray
    epsilon: float
    n_samples: int

    def most_sensitive(self, n: int = 5) -> list[tuple[str, float]]:
        """
        Return the top-n most sensitive features as (name, score) pairs.

        Parameters
        ----------
        n : int
            Number of features to return. Default 5.
        """
        idx = np.argsort(self.scores)[::-1][:n]
        return [(self.feature_names[i], float(self.scores[i])) for i in idx]

    def __str__(self) -> str:
        lines = [
            f"SensitivityResult  epsilon={self.epsilon}  n_samples={self.n_samples}"
        ]
        ranked = sorted(
            zip(self.feature_names, self.scores), key=lambda x: -x[1]
        )
        for name, score in ranked[:10]:
            bar = "#" * min(int(score * 40), 40)
            lines.append(f"  {name:<20} {score:.4f}  {bar}")
        if len(self.feature_names) > 10:
            lines.append(f"  ... ({len(self.feature_names) - 10} more feature(s))")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SensitivityResult(n_features={len(self.feature_names)}, "
            f"epsilon={self.epsilon}, n_samples={self.n_samples})"
        )


def encoding_sensitivity(
    encoder,
    dataset,
    epsilon: float = 0.01,
    n_samples: int = 20,
    seed: int = 42,
) -> SensitivityResult:
    """
    Measure how much each feature influences the encoded quantum state.

    Perturbs each feature independently by *epsilon* and measures the mean
    state infidelity ``(1 − |⟨ψ|ψ'⟩|²)`` across *n_samples* data points.
    Features with higher scores have more influence on the circuit output —
    useful for debugging encodings and identifying which features the quantum
    model is most sensitive to.

    Only works for encodings supported by the numpy statevector simulator
    (``n_qubits ≤ 12``). Returns zero scores for unsupported encodings or
    when simulation fails.

    Parameters
    ----------
    encoder : BaseEncoder
        Fitted encoder instance.
    dataset : Dataset
        Dataset to sample from.
    epsilon : float
        Perturbation magnitude (absolute, in the feature's current scale).
        Default 0.01.
    n_samples : int
        Number of dataset samples to average over. Default 20.
    seed : int
        Random seed for sample selection. Default 42.

    Returns
    -------
    SensitivityResult
    """
    from quprep.metrics._simulate import statevector_from_encoded

    feature_names = (
        list(dataset.feature_names)
        if dataset.feature_names
        else [f"feature[{i}]" for i in range(dataset.n_features)]
    )
    n_features = dataset.n_features
    scores = np.zeros(n_features)

    rng = np.random.default_rng(seed)
    n = min(n_samples, dataset.n_samples)
    sample_idx = rng.choice(dataset.n_samples, n, replace=False)
    samples = dataset.data[sample_idx]

    for j in range(n_features):
        diffs = []
        for x in samples:
            if np.isnan(x).any():
                continue
            try:
                e_orig = encoder.encode(x)
                sv_orig = statevector_from_encoded(e_orig)
                if sv_orig is None:
                    continue
                x_pert = x.copy()
                x_pert[j] += epsilon
                e_pert = encoder.encode(x_pert)
                sv_pert = statevector_from_encoded(e_pert)
                if sv_pert is None:  # pragma: no cover
                    continue
                fidelity = float(abs(np.dot(sv_orig.conj(), sv_pert)) ** 2)
                diffs.append(1.0 - fidelity)
            except Exception:
                continue
        scores[j] = float(np.mean(diffs)) if diffs else 0.0

    return SensitivityResult(
        feature_names=feature_names,
        scores=scores,
        epsilon=epsilon,
        n_samples=n,
    )
