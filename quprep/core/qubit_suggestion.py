"""Auto qubit count suggestion based on dataset profile."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

# Practical NISQ execution ceiling — above this, most current NISQ hardware struggles
_NISQ_CEILING = 20


@dataclass
class QubitSuggestion:
    """
    Qubit budget recommendation for a dataset.

    Attributes
    ----------
    n_qubits : int
        Recommended qubit count.
    n_features : int
        Number of features in the dataset (before any reduction).
    nisq_safe : bool
        ``True`` if ``n_qubits <= 20`` (practical NISQ ceiling).
    encoding_hint : str
        Encoding that works well at this qubit count and task.
    reasoning : str
        Human-readable explanation of the recommendation.
    warning : str or None
        Warning if dimensionality reduction is strongly recommended.
    """

    n_qubits: int
    n_features: int
    nisq_safe: bool
    encoding_hint: str
    reasoning: str
    warning: str | None

    def __str__(self) -> str:
        lines = [
            f"Suggested qubits : {self.n_qubits}",
            f"Dataset features : {self.n_features}",
            f"NISQ-safe        : {'yes' if self.nisq_safe else 'NO'}",
            f"Encoding hint    : {self.encoding_hint}",
            f"Reasoning        : {self.reasoning}",
        ]
        if self.warning:
            lines.append(f"Warning          : {self.warning}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"QubitSuggestion(n_qubits={self.n_qubits}, "
            f"encoding_hint='{self.encoding_hint}', "
            f"nisq_safe={self.nisq_safe})"
        )


def suggest_qubits(
    source,
    *,
    task: str = "classification",
    max_qubits: int | None = None,
) -> QubitSuggestion:
    """
    Suggest an appropriate qubit budget for a dataset.

    Analyses the dataset's feature count and sample count to recommend a
    qubit count that is practical on NISQ hardware. For datasets with more
    features than the budget allows, a dimensionality reduction step is
    recommended.

    Parameters
    ----------
    source : str, Path, np.ndarray, pd.DataFrame, or Dataset
        Input data.
    task : str
        Target task: 'classification', 'regression', 'qaoa', 'kernel',
        'simulation'. Influences the encoding hint.
    max_qubits : int, optional
        Hard upper bound on the suggestion. Defaults to 20 (practical NISQ
        ceiling).

    Returns
    -------
    QubitSuggestion
    """
    _valid_tasks = {"classification", "regression", "qaoa", "kernel", "simulation"}
    if task not in _valid_tasks:
        raise ValueError(
            f"Unknown task '{task}'. Choose from: {sorted(_valid_tasks)}"
        )

    dataset = _ingest(source)
    d = dataset.n_features
    n = dataset.n_samples
    ceiling = max_qubits if max_qubits is not None else _NISQ_CEILING

    # --- Qubit count and size reasoning ---
    if d <= ceiling:
        n_qubits = d
        warning = None
        size_reason = (
            f"dataset has {d} feature(s) — one qubit per feature fits within "
            f"the {'specified' if max_qubits is not None else 'NISQ'} budget of {ceiling}"
        )
    else:
        n_qubits = ceiling
        warning = (
            f"Dataset has {d} features but qubit budget is {ceiling}. "
            f"Apply a reducer (e.g. PCAReducer(n_components={ceiling})) "
            f"before encoding to avoid information loss."
        )
        size_reason = (
            f"dataset has {d} features; capped at {ceiling} qubits — "
            f"apply dimensionality reduction first"
        )

    # --- Encoding hint ---
    amp_qubits = max(1, math.ceil(math.log2(max(d, 2))))

    if task == "qaoa":
        hint = "basis"
        hint_reason = "basis encoding maps naturally to QAOA binary variables"
    elif task == "kernel":
        if n_qubits <= 8:
            hint = "iqp"
            hint_reason = (
                "IQP encoding is ideal for kernel methods at this qubit count"
            )
        else:
            hint = "angle"
            hint_reason = (
                "angle encoding preferred for kernel tasks with many qubits "
                "(IQP depth grows as O(d²))"
            )
    elif task == "simulation":
        hint = "hamiltonian"
        hint_reason = "Hamiltonian encoding directly represents physical time evolution"
    elif n > 500:
        hint = "angle"
        hint_reason = (
            "angle encoding scales to large sample counts; "
            "amplitude encoding requires per-sample state preparation"
        )
    elif n_qubits <= 4 and n <= 100 and amp_qubits <= n_qubits:
        hint = "amplitude"
        hint_reason = (
            "amplitude encoding is feasible for small qubit counts and sample sizes"
        )
    else:
        hint = "angle"
        hint_reason = "angle encoding is NISQ-safe and widely applicable"

    nisq_safe = n_qubits <= _NISQ_CEILING
    reasoning = f"{size_reason}; {hint_reason}"

    return QubitSuggestion(
        n_qubits=n_qubits,
        n_features=d,
        nisq_safe=nisq_safe,
        encoding_hint=hint,
        reasoning=reasoning,
        warning=warning,
    )


def _ingest(source):
    """Return a Dataset from any supported source type."""
    from quprep.core.dataset import Dataset

    if isinstance(source, Dataset):
        return source

    import numpy as np

    if isinstance(source, (str, Path)):
        from quprep.ingest.csv_ingester import CSVIngester
        return CSVIngester().load(source)

    if isinstance(source, (np.ndarray, list)):
        from quprep.ingest.numpy_ingester import NumpyIngester
        return NumpyIngester().load(source)

    try:
        import pandas as pd
        if isinstance(source, pd.DataFrame):
            from quprep.ingest.numpy_ingester import NumpyIngester
            return NumpyIngester().load(source)
    except ImportError:
        pass

    raise TypeError(
        f"Cannot ingest source of type '{type(source).__name__}'. "
        "Pass a file path, np.ndarray, pd.DataFrame, or Dataset."
    )
