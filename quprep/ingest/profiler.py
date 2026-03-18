"""Dataset profiling — statistics, distributions, feature types, class balance."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quprep.core.dataset import Dataset


@dataclass
class DatasetProfile:
    """
    Summary statistics computed during ingestion.

    Attributes
    ----------
    n_samples : int
    n_features : int
    feature_names : list of str
    feature_types : list of str
        Per-feature type: 'continuous', 'discrete', 'binary', 'categorical'.
    missing_counts : np.ndarray
        Number of missing values per feature.
    means : np.ndarray
    stds : np.ndarray
    class_balance : dict or None
        Class frequencies for classification datasets.
    """

    n_samples: int = 0
    n_features: int = 0
    feature_names: list[str] = field(default_factory=list)
    feature_types: list[str] = field(default_factory=list)
    missing_counts: np.ndarray | None = None
    means: np.ndarray | None = None
    stds: np.ndarray | None = None
    class_balance: dict | None = None


def profile(dataset: Dataset) -> DatasetProfile:
    """Compute a DatasetProfile for the given Dataset."""
    raise NotImplementedError("profile() — coming in v0.1.0")
