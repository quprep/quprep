"""Encoding recommendation engine."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Encoding profiles
# ---------------------------------------------------------------------------

_ENCODINGS: dict[str, dict] = {
    "angle": {
        "nisq_safe": True,
        "depth": "O(d)",
        "qubit_fn": lambda d: d,
        "task_scores": {
            "classification": 8,
            "regression": 8,
            "qaoa": 4,
            "kernel": 5,
            "simulation": 3,
        },
        "continuous_bonus": 2,
        "binary_bonus": 0,
        "description": (
            "Single-qubit rotation per feature. Shallow, general-purpose, NISQ-safe."
        ),
    },
    "amplitude": {
        "nisq_safe": False,
        "depth": "O(2^d)",
        "qubit_fn": lambda d: max(1, int(np.ceil(np.log2(max(d, 2))))),
        "task_scores": {
            "classification": 5,
            "regression": 6,
            "qaoa": 2,
            "kernel": 4,
            "simulation": 4,
        },
        "continuous_bonus": 3,
        "binary_bonus": 0,
        "description": (
            "Entire vector as quantum amplitudes. Qubit-efficient but exponential depth."
        ),
    },
    "basis": {
        "nisq_safe": True,
        "depth": "O(d)",
        "qubit_fn": lambda d: d,
        "task_scores": {
            "classification": 3,
            "regression": 1,
            "qaoa": 10,
            "kernel": 2,
            "simulation": 1,
        },
        "continuous_bonus": 0,
        "binary_bonus": 5,
        "description": (
            "Binary feature map via X gates. Shallowest possible, ideal for QAOA."
        ),
    },
    "iqp": {
        "nisq_safe": True,
        "depth": "O(d²·reps)",
        "qubit_fn": lambda d: d,
        "task_scores": {
            "classification": 9,
            "regression": 5,
            "qaoa": 3,
            "kernel": 10,
            "simulation": 5,
        },
        "continuous_bonus": 2,
        "binary_bonus": 0,
        "description": (
            "Havlíček 2019 feature map with pairwise interactions. "
            "Best for kernel methods and correlated features."
        ),
    },
    "reupload": {
        "nisq_safe": True,
        "depth": "O(d·layers)",
        "qubit_fn": lambda d: d,
        "task_scores": {
            "classification": 8,
            "regression": 9,
            "qaoa": 3,
            "kernel": 7,
            "simulation": 4,
        },
        "continuous_bonus": 2,
        "binary_bonus": 0,
        "description": (
            "Pérez-Salinas 2020 data re-uploading. "
            "Universal approximation, best for large datasets."
        ),
    },
    "entangled_angle": {
        "nisq_safe": True,
        "depth": "O(d·layers + CNOT layers)",
        "qubit_fn": lambda d: d,
        "task_scores": {
            "classification": 7,
            "regression": 7,
            "qaoa": 3,
            "kernel": 8,
            "simulation": 4,
        },
        "continuous_bonus": 2,
        "binary_bonus": 0,
        "description": (
            "Angle encoding with CNOT entanglement layers. "
            "Captures feature correlations; good for kernel methods."
        ),
    },
    "hamiltonian": {
        "nisq_safe": False,
        "depth": "O(d·steps)",
        "qubit_fn": lambda d: d,
        "task_scores": {
            "classification": 3,
            "regression": 4,
            "qaoa": 2,
            "kernel": 4,
            "simulation": 10,
        },
        "continuous_bonus": 2,
        "binary_bonus": 0,
        "description": (
            "Trotterized Z Hamiltonian evolution. "
            "Designed for physics simulation / VQE."
        ),
    },
    "zz_feature_map": {
        "nisq_safe": True,
        "depth": "O(d²·reps)",
        "qubit_fn": lambda d: d,
        "task_scores": {
            "classification": 9,
            "regression": 5,
            "qaoa": 2,
            "kernel": 10,
            "simulation": 4,
        },
        "continuous_bonus": 2,
        "binary_bonus": 0,
        "description": (
            "Havlíček ZZ feature map with pairwise ZZ interactions. "
            "Gold standard for quantum kernel methods."
        ),
    },
    "pauli_feature_map": {
        "nisq_safe": True,
        "depth": "O(d²·reps)",
        "qubit_fn": lambda d: d,
        "task_scores": {
            "classification": 8,
            "regression": 5,
            "qaoa": 2,
            "kernel": 9,
            "simulation": 5,
        },
        "continuous_bonus": 2,
        "binary_bonus": 0,
        "description": (
            "Configurable Pauli string feature map. "
            "More flexible than ZZFeatureMap; good for kernel methods."
        ),
    },
    "random_fourier": {
        "nisq_safe": True,
        "depth": "O(n_components)",
        "qubit_fn": lambda d: d,
        "task_scores": {
            "classification": 8,
            "regression": 9,
            "qaoa": 1,
            "kernel": 9,
            "simulation": 2,
        },
        "continuous_bonus": 3,
        "binary_bonus": 0,
        "description": (
            "Random Fourier features approximating an RBF kernel. "
            "Requires fit(); strong for regression and large continuous datasets."
        ),
    },
    "tensor_product": {
        "nisq_safe": True,
        "depth": "O(d)",
        "qubit_fn": lambda d: d,
        "task_scores": {
            "classification": 7,
            "regression": 7,
            "qaoa": 3,
            "kernel": 6,
            "simulation": 3,
        },
        "continuous_bonus": 2,
        "binary_bonus": 0,
        "description": (
            "Ry+Rz per qubit — full Bloch sphere coverage with no entanglement. "
            "More expressive than angle encoding, same depth."
        ),
    },
    "qaoa_problem": {
        "nisq_safe": True,
        "depth": "O(d·p)",
        "qubit_fn": lambda d: d,
        "task_scores": {
            "classification": 4,
            "regression": 3,
            "qaoa": 10,
            "kernel": 4,
            "simulation": 5,
        },
        "continuous_bonus": 1,
        "binary_bonus": 2,
        "description": (
            "QAOA-inspired feature map. Specifically designed for quantum "
            "optimization workflows; not a general QML encoder."
        ),
    },
}

_VALID_TASKS = frozenset(_ENCODINGS["angle"]["task_scores"].keys())


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EncodingRecommendation:
    """
    Result of the encoding recommendation engine.

    Attributes
    ----------
    method : str
        Recommended encoding method (e.g. ``'iqp'``).
    qubits : int
        Number of qubits required for the dataset's feature count.
    depth : str
        Asymptotic circuit depth expression.
    nisq_safe : bool
        Whether the encoding is suitable for current NISQ devices.
    reason : str
        Human-readable explanation of the recommendation.
    score : float
        Internal score (higher is better). For comparison only.
    alternatives : list[EncodingRecommendation]
        Runner-up options ranked by score, without nested alternatives.
    """

    method: str
    qubits: int
    depth: str
    nisq_safe: bool
    reason: str
    score: float
    alternatives: list[EncodingRecommendation] = field(default_factory=list)

    def apply(self, source, *, framework: str = "qasm", **kwargs):
        """Apply this recommendation to source data and export circuits.

        Parameters
        ----------
        source : str, Path, np.ndarray, or pd.DataFrame
            Input data.
        framework : str
            Export target. Default ``'qasm'``.
        **kwargs
            Forwarded to :func:`quprep.prepare`.

        Returns
        -------
        PipelineResult
        """
        import quprep
        return quprep.prepare(source, encoding=self.method, framework=framework, **kwargs)

    def __str__(self) -> str:
        lines = [
            f"Recommended encoding : {self.method}",
            f"Qubits needed        : {self.qubits}",
            f"Circuit depth        : {self.depth}",
            f"NISQ safe            : {'yes' if self.nisq_safe else 'no'}",
            f"Score                : {self.score:.1f}",
            f"Reason               : {self.reason}",
        ]
        if self.alternatives:
            lines.append("Alternatives         :")
            for alt in self.alternatives:
                lines.append(
                    f"  {alt.method:<16} score={alt.score:.1f}  {alt.depth}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score(
    encoding: str,
    profile: dict,
    task: str,
    qubit_budget: int | None,
) -> float:
    """Return a score for an encoding given dataset profile and task."""
    enc = _ENCODINGS[encoding]
    d = profile["n_features"]
    n_samples = profile["n_samples"]
    missing_rate = profile["missing_rate"]
    sparsity = profile["sparsity"]
    has_negatives = profile["has_negatives"]
    feature_collinear = profile["feature_collinear"]

    # Base: task fit (0-10) × 5
    score = enc["task_scores"][task] * 5.0

    # Data type bonus (binary/continuous fraction)
    score += enc["binary_bonus"] * profile["binary_fraction"]
    score += enc["continuous_bonus"] * profile["continuous_fraction"]

    # NISQ bonus (small but meaningful tiebreaker)
    if enc["nisq_safe"]:
        score += 3.0

    # ----------------------------------------------------------------
    # Dataset-aware adjustments
    # ----------------------------------------------------------------

    if encoding == "amplitude":
        # Amplitude state prep is expensive per sample — penalise large datasets
        if n_samples > 5000:
            score -= 15.0
        elif n_samples > 1000:
            score -= 8.0
        elif n_samples > 500:
            score -= 4.0

        # Amplitude naturally handles negative values via superposition
        if has_negatives:
            score += 2.0

        # High missing rate is dangerous: amplitude requires exact unit norm
        if missing_rate > 0.2:
            score -= 8.0
        elif missing_rate > 0.1:
            score -= 4.0

    if encoding == "basis":
        # Sparse data (many exact zeros) aligns naturally with basis encoding
        if sparsity > 0.3:
            score += 3.0 * sparsity
        elif sparsity > 0.1:
            score += 1.5 * sparsity

        # Negative values: basis binarizes at 0.5, so all negatives map to 0 → info loss
        if has_negatives:
            score -= 4.0

    if encoding in ("iqp", "entangled_angle"):
        # Correlated features: entanglement captures inter-feature relationships
        if feature_collinear:
            score += 4.0

        # IQP depth grows as O(d²) — penalise wide datasets
        if encoding == "iqp" and d > 15:
            score -= (d - 15) * 0.4

    if encoding == "reupload":
        # High expressivity can overfit with very few samples
        if n_samples < 20:
            score -= 8.0
        elif n_samples < 50:
            score -= 4.0

        # Conversely, re-uploading shines on large datasets (universal approximation)
        if n_samples > 500:
            score += 3.0

    if encoding in ("zz_feature_map", "pauli_feature_map"):
        # Correlated features: ZZ/Pauli interactions capture inter-feature relationships
        if feature_collinear:
            score += 4.0

        # O(d²) depth — penalise wide datasets
        if d > 15:
            score -= (d - 15) * 0.4

    if encoding == "random_fourier":
        # RBF approximation is poor on binary data — rotations assume continuous input
        if profile["binary_fraction"] > 0.5:
            score -= 6.0

        # Shines on large continuous datasets (approximation quality improves with data)
        if n_samples > 500 and profile["continuous_fraction"] > 0.5:
            score += 3.0

        # Requires fit() — penalise if no training data (very small datasets)
        if n_samples < 30:
            score -= 5.0

    if encoding == "qaoa_problem":
        # Heavily penalise non-QAOA tasks — this is a specialist encoder
        if task != "qaoa":
            score -= 10.0

    # ----------------------------------------------------------------
    # Qubit budget
    # ----------------------------------------------------------------
    if qubit_budget is not None:
        needed = enc["qubit_fn"](d)
        if needed > qubit_budget:
            score -= 40.0
        elif encoding == "amplitude" and needed <= qubit_budget:
            # Amplitude is qubit-efficient — reward when within budget
            score += 5.0

    return score


def _build_reason(
    encoding: str,
    profile: dict,
    task: str,
    score: float,
    qubit_budget: int | None,
) -> str:
    enc = _ENCODINGS[encoding]
    d = profile["n_features"]
    n_samples = profile["n_samples"]
    missing_rate = profile["missing_rate"]
    sparsity = profile["sparsity"]
    has_negatives = profile["has_negatives"]
    feature_collinear = profile["feature_collinear"]
    parts = []

    task_score = enc["task_scores"][task]
    if task_score >= 9:
        parts.append(f"best fit for {task} tasks")
    elif task_score >= 7:
        parts.append(f"good fit for {task} tasks")
    else:
        parts.append(f"acceptable for {task} tasks")

    binary_frac = profile["binary_fraction"]
    continuous_frac = profile["continuous_fraction"]
    if enc["binary_bonus"] > 0 and binary_frac > 0.5:
        parts.append(f"{binary_frac:.0%} binary features favour this encoding")
    if enc["continuous_bonus"] > 0 and continuous_frac > 0.5:
        parts.append("continuous features map naturally to rotation angles")

    if enc["nisq_safe"]:
        parts.append("NISQ-safe (shallow circuit)")
    else:
        parts.append("deep circuit — fault-tolerant hardware recommended")

    # Dataset-specific observations
    if encoding == "amplitude":
        if n_samples > 1000:
            parts.append(
                f"note: {n_samples} samples — state prep overhead is high; "
                "consider angle or reupload for large datasets"
            )
        if has_negatives:
            parts.append("negative values handled naturally via superposition")
        if missing_rate > 0.1:
            parts.append(
                f"warning: {missing_rate:.0%} missing values — impute before encoding"
            )

    if encoding == "basis" and sparsity > 0.3:
        parts.append(f"{sparsity:.0%} near-zero values align naturally with X-gate encoding")

    if encoding in ("iqp", "entangled_angle") and feature_collinear:
        parts.append("correlated features benefit from entanglement structure")

    if encoding in ("zz_feature_map", "pauli_feature_map"):
        if feature_collinear:
            parts.append("correlated features benefit from ZZ/Pauli interaction structure")
        if d > 15:
            parts.append(
                f"note: {d} features — O(d²) depth may be costly; "
                "consider angle or reupload for very wide datasets"
            )

    if encoding == "random_fourier":
        if profile["binary_fraction"] > 0.5:
            parts.append(
                "warning: majority binary features — RBF approximation works best "
                "on continuous data; consider basis or angle encoding instead"
            )
        if n_samples > 500:
            parts.append(
                f"{n_samples} samples — RBF kernel approximation improves with more data"
            )
        parts.append("note: requires fit() before transform()")

    if encoding == "qaoa_problem" and task != "qaoa":
        parts.append(
            "note: QAOAProblemEncoder is designed for quantum optimization workflows; "
            "for general QML use angle, iqp, or zz_feature_map instead"
        )

    if encoding == "reupload":
        if n_samples > 500:
            parts.append(
                f"{n_samples} samples — re-uploading layers improve expressivity"
            )
        elif n_samples < 50:
            parts.append(
                f"only {n_samples} samples — monitor for overfitting with deep circuits"
            )

    if qubit_budget is not None:
        needed = enc["qubit_fn"](d)
        parts.append(f"needs {needed} qubit(s) (budget: {qubit_budget})")

    return "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recommend(
    source,
    *,
    task: str = "classification",
    qubits: int | None = None,
    **kwargs,
) -> EncodingRecommendation:
    """
    Recommend the best encoding for a dataset and task.

    Scores all encodings against the dataset profile (feature count, binary/
    continuous fraction, missing rate, sparsity, correlations, sample count)
    and the target task, then returns the highest-scoring option with ranked
    alternatives.

    Parameters
    ----------
    source : str, Path, np.ndarray, pd.DataFrame, or Dataset
        Input data. Accepts anything the pipeline ingester accepts.
    task : str
        Target task: ``'classification'``, ``'regression'``, ``'qaoa'``,
        ``'kernel'``, or ``'simulation'``. Default ``'classification'``.
    qubits : int, optional
        Maximum qubit budget. Encodings that exceed this are heavily penalised.
    **kwargs
        Reserved for future use (e.g. ``backend='ibm_brisbane'``).

    Returns
    -------
    EncodingRecommendation
        Top recommendation with alternatives list.

    Raises
    ------
    ValueError
        If ``task`` is not one of the supported values.
    """
    if task not in _VALID_TASKS:
        raise ValueError(
            f"Unknown task '{task}'. Choose from: {sorted(_VALID_TASKS)}"
        )

    # Ingest and profile
    profile = _profile_source(source)

    # Score all encodings
    scored = []
    for name in _ENCODINGS:
        s = _score(name, profile, task, qubits)
        scored.append((name, s))
    scored.sort(key=lambda x: x[1], reverse=True)

    # Build result objects (no nested alternatives)
    def _make(name: str, s: float) -> EncodingRecommendation:
        enc = _ENCODINGS[name]
        d = profile["n_features"]
        return EncodingRecommendation(
            method=name,
            qubits=enc["qubit_fn"](d),
            depth=enc["depth"],
            nisq_safe=enc["nisq_safe"],
            reason=_build_reason(name, profile, task, s, qubits),
            score=round(s, 1),
        )

    best_name, best_score = scored[0]
    alternatives = [_make(name, s) for name, s in scored[1:]]
    top = _make(best_name, best_score)
    top.alternatives = alternatives
    return top


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _profile_source(source) -> dict:
    """Ingest source and return a lightweight profile dict."""
    from quprep.core.dataset import Dataset
    from quprep.ingest.csv_ingester import CSVIngester
    from quprep.ingest.numpy_ingester import NumpyIngester

    try:
        import pandas as pd
        _has_pandas = True
    except ImportError:
        _has_pandas = False

    if isinstance(source, Dataset):
        dataset = source
    elif isinstance(source, str) or hasattr(source, "__fspath__"):
        dataset = CSVIngester().load(str(source))
    elif _has_pandas and isinstance(source, pd.DataFrame):
        dataset = NumpyIngester().load(source)
    else:
        dataset = NumpyIngester().load(source)

    data = dataset.data
    n_samples, n_features = data.shape

    # Binary fraction: features where all non-NaN values are in {0, 1}
    binary_cols = 0
    for j in range(n_features):
        col = data[:, j]
        col_valid = col[~np.isnan(col)]
        if len(col_valid) > 0 and np.all(np.isin(col_valid, [0.0, 1.0])):
            binary_cols += 1
    binary_fraction = binary_cols / n_features if n_features > 0 else 0.0
    continuous_fraction = 1.0 - binary_fraction

    # Missing rate: fraction of NaN values across the full matrix
    missing_rate = float(np.mean(np.isnan(data)))

    # Sparsity: fraction of values that are exactly zero
    non_nan = data[~np.isnan(data)]
    sparsity = float(np.mean(non_nan == 0.0)) if len(non_nan) > 0 else 0.0

    # Has negatives: any value strictly below zero
    has_negatives = bool(np.any(non_nan < 0.0)) if len(non_nan) > 0 else False

    # Feature collinearity: mean absolute pairwise Pearson correlation
    # Only computed for manageable feature counts (≤ 50); defaults to False otherwise.
    feature_collinear = False
    if 2 <= n_features <= 50 and n_samples >= 5:
        try:
            # Drop columns that are all-NaN before computing correlation
            valid_mask = ~np.all(np.isnan(data), axis=0)
            clean = data[:, valid_mask]
            if clean.shape[1] >= 2:
                corr = np.corrcoef(clean, rowvar=False)
                # Upper triangle excluding diagonal
                idx = np.triu_indices(corr.shape[0], k=1)
                mean_abs_corr = float(np.nanmean(np.abs(corr[idx])))
                feature_collinear = mean_abs_corr > 0.3
        except Exception:
            pass

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "binary_fraction": binary_fraction,
        "continuous_fraction": continuous_fraction,
        "missing_rate": missing_rate,
        "sparsity": sparsity,
        "has_negatives": has_negatives,
        "feature_collinear": feature_collinear,
    }
