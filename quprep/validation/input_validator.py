"""Input validation utilities and warning types for pipeline entry."""

from __future__ import annotations

import warnings

import numpy as np

from quprep.core.dataset import Dataset


class QuPrepWarning(UserWarning):
    """Base warning class for all QuPrep pipeline warnings."""


def validate_dataset(dataset: Dataset, *, context: str = "") -> None:
    """
    Validate a Dataset for structural problems.

    Raises ValueError for hard failures; emits QuPrepWarning for
    recoverable issues like NaN values.

    Parameters
    ----------
    dataset : Dataset
    context : str
        Optional stage name for clearer messages (e.g. ``'after Imputer'``).

    Raises
    ------
    ValueError
        If ``dataset.data`` is not 2-D, has zero samples, zero features,
        or is a non-float dtype.
    """
    where = f" {context}" if context else ""
    data = dataset.data

    if data.ndim != 2:
        raise ValueError(
            f"Dataset.data must be 2-D, got shape {data.shape}{where}."
        )
    if data.shape[0] == 0:
        raise ValueError(
            f"Dataset has no samples (n_samples=0){where}."
        )
    if data.shape[1] == 0:
        raise ValueError(
            f"Dataset has no features (n_features=0){where}. "
            "Check your ingestion step."
        )
    if not np.issubdtype(data.dtype, np.floating):
        raise ValueError(
            f"Dataset.data dtype is {data.dtype!r}{where}; expected float64. "
            "Pass numeric data or use an Imputer/CategoricalEncoder."
        )

    nan_cols = int(np.isnan(data).any(axis=0).sum())
    if nan_cols:
        nan_frac = float(np.isnan(data).mean())
        warnings.warn(
            f"Dataset contains NaN in {nan_cols} of {data.shape[1]} features "
            f"(overall fraction {nan_frac:.1%}){where}. "
            "Add an Imputer to your pipeline.",
            QuPrepWarning,
            stacklevel=3,
        )


def warn_qubit_mismatch(n_features: int, n_qubits: int, encoding: str) -> None:
    """
    Warn if the dataset has more features than the configured qubit budget.

    Parameters
    ----------
    n_features : int
        Number of features in the dataset.
    n_qubits : int
        User-specified qubit budget.
    encoding : str
        Name of the encoding method.
    """
    if n_features > n_qubits:
        warnings.warn(
            f"{encoding} maps {n_features} features to {n_features} qubits, "
            f"but your qubit budget is {n_qubits} — information will be lost "
            "unless you add a reducer (e.g. PCAReducer or HardwareAwareReducer).",
            QuPrepWarning,
            stacklevel=3,
        )
