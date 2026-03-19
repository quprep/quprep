"""Internal dataset representation used across pipeline stages."""

from __future__ import annotations

import numpy as np


class Dataset:
    """
    Internal representation of a dataset as it flows through the pipeline.

    Wraps numeric data alongside metadata and any non-numeric (categorical)
    columns that haven't been encoded yet. All pipeline stages operate on
    Dataset objects rather than raw arrays or frames.

    Parameters
    ----------
    data : np.ndarray
        Numeric feature matrix, shape (n_samples, n_features). float64.
    feature_names : list of str, optional
        Column names for the numeric features in `data`.
    feature_types : list of str, optional
        Detected type per numeric feature: 'continuous', 'discrete', 'binary'.
    categorical_data : dict of {str: list}, optional
        Non-numeric columns not yet encoded. Keys are column names, values
        are lists of raw values. CategoricalEncoder moves columns from here
        into `data` once they are encoded.
    metadata : dict, optional
        Arbitrary metadata (source path, provenance, etc.).
    """

    def __init__(
        self,
        data: np.ndarray,
        feature_names: list[str] | None = None,
        feature_types: list[str] | None = None,
        categorical_data: dict[str, list] | None = None,
        metadata: dict | None = None,
    ):
        self.data = data
        self.feature_names = feature_names or []
        self.feature_types = feature_types or []
        self.categorical_data = categorical_data or {}
        self.metadata = metadata or {}

    @property
    def n_samples(self) -> int:
        return self.data.shape[0]

    @property
    def n_features(self) -> int:
        return self.data.shape[1]

    @property
    def n_categorical(self) -> int:
        return len(self.categorical_data)

    def __repr__(self) -> str:
        cat = f", categorical={self.n_categorical}" if self.categorical_data else ""
        return f"Dataset(n_samples={self.n_samples}, n_features={self.n_features}{cat})"
