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
    mins : np.ndarray
    maxs : np.ndarray
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
    mins: np.ndarray | None = None
    maxs: np.ndarray | None = None
    class_balance: dict | None = None

    def __str__(self) -> str:
        lines = [
            f"DatasetProfile",
            f"  samples   : {self.n_samples}",
            f"  features  : {self.n_features}",
        ]
        if self.feature_names:
            lines.append(f"  names     : {self.feature_names}")
        if self.feature_types:
            counts = {}
            for t in self.feature_types:
                counts[t] = counts.get(t, 0) + 1
            lines.append(f"  types     : {counts}")
        if self.missing_counts is not None:
            total_missing = int(self.missing_counts.sum())
            lines.append(f"  missing   : {total_missing} total")
        return "\n".join(lines)


def profile(dataset: Dataset) -> DatasetProfile:
    """
    Compute a DatasetProfile for a Dataset.

    Parameters
    ----------
    dataset : Dataset

    Returns
    -------
    DatasetProfile
    """
    data = dataset.data
    n_samples, n_features = data.shape

    missing_counts = np.isnan(data).sum(axis=0)

    with np.errstate(invalid="ignore"):
        means = np.nanmean(data, axis=0)
        stds = np.nanstd(data, axis=0)
        mins = np.nanmin(data, axis=0)
        maxs = np.nanmax(data, axis=0)

    return DatasetProfile(
        n_samples=n_samples,
        n_features=n_features,
        feature_names=list(dataset.feature_names),
        feature_types=list(dataset.feature_types),
        missing_counts=missing_counts,
        means=means,
        stds=stds,
        mins=mins,
        maxs=maxs,
    )
