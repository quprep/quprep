"""Internal dataset representation used across pipeline stages."""

from __future__ import annotations

import numpy as np


class Dataset:
    """
    Internal representation of a dataset as it flows through the pipeline.

    Wraps the raw data alongside metadata derived during ingestion: feature
    types, statistics, and provenance information. All pipeline stages operate
    on Dataset objects rather than raw arrays or frames.

    Parameters
    ----------
    data : np.ndarray
        Numeric feature matrix, shape (n_samples, n_features).
    feature_names : list of str, optional
        Column names.
    feature_types : list of str, optional
        Detected type per feature: 'continuous', 'discrete', 'binary', 'categorical'.
    metadata : dict, optional
        Arbitrary metadata (source path, profiling stats, etc.).
    """

    def __init__(
        self,
        data: np.ndarray,
        feature_names: list[str] | None = None,
        feature_types: list[str] | None = None,
        metadata: dict | None = None,
    ):
        self.data = data
        self.feature_names = feature_names or []
        self.feature_types = feature_types or []
        self.metadata = metadata or {}

    @property
    def n_samples(self) -> int:
        return self.data.shape[0]

    @property
    def n_features(self) -> int:
        return self.data.shape[1]

    def __repr__(self) -> str:
        return f"Dataset(n_samples={self.n_samples}, n_features={self.n_features})"
