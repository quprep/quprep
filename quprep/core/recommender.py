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
        "description": "Single-qubit rotation per feature. Shallow, general-purpose, NISQ-safe.",
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
        "description": "Entire vector as quantum amplitudes. Qubit-efficient but exponential depth.",  # noqa: E501
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
        "description": "Binary feature map via X gates. Shallowest possible, ideal for QAOA.",
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
        "description": "Havlíček 2019 feature map with pairwise interactions. Best for kernel methods.",  # noqa: E501
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
        "description": "Pérez-Salinas 2020 data re-uploading. Universal approximation, high expressivity.",  # noqa: E501
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
        "description": "Trotterized Z Hamiltonian evolution. Designed for physics simulation / VQE.",  # noqa: E501
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
                    f"  {alt.method:<12} score={alt.score:.1f}  {alt.depth}"
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
    """Return a 0-100 score for an encoding given dataset profile and task."""
    enc = _ENCODINGS[encoding]
    d = profile["n_features"]

    # Base: task fit (0-10) × 5
    score = enc["task_scores"][task] * 5.0

    # Data type bonus
    binary_frac = profile["binary_fraction"]
    continuous_frac = profile["continuous_fraction"]
    score += enc["binary_bonus"] * binary_frac
    score += enc["continuous_bonus"] * continuous_frac

    # NISQ bonus (small but meaningful tiebreaker)
    if enc["nisq_safe"]:
        score += 3.0

    # Qubit budget: penalise if this encoding needs more qubits than available
    if qubit_budget is not None:
        needed = enc["qubit_fn"](d)
        if needed > qubit_budget:
            # Hard over-budget: heavy penalty
            score -= 40.0
        elif encoding == "amplitude" and needed == enc["qubit_fn"](d):
            # Amplitude is qubit-efficient — reward within budget
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

    if qubit_budget is not None:
        needed = enc["qubit_fn"](profile["n_features"])
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

    Scores all encodings against the dataset profile (feature count, binary
    fraction, continuous fraction) and the target task, then returns the
    highest-scoring option with ranked alternatives.

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

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "binary_fraction": binary_fraction,
        "continuous_fraction": continuous_fraction,
    }
