"""Missing value imputation strategies."""

from __future__ import annotations

import numpy as np

from quprep.core.dataset import Dataset

_VALID_STRATEGIES = ("mean", "median", "mode", "knn", "mice", "drop")


class Imputer:
    """
    Handle missing values in a Dataset.

    Parameters
    ----------
    strategy : str
        'mean'   — replace NaN with column mean.
        'median' — replace NaN with column median.
        'mode'   — replace NaN with column mode (most frequent value).
        'knn'    — k-nearest neighbours imputation (sklearn).
        'mice'   — iterative/chained-equations imputation (sklearn).
        'drop'   — drop rows that still contain NaN after column pruning.
    drop_threshold : float
        Drop an entire feature column if more than this fraction of its
        values are missing. Applied before row-level strategies.
        Default 0.5 (drop columns missing more than 50% of values).
    knn_neighbors : int
        Number of neighbours for KNN imputation. Default 5.
    """

    def __init__(
        self,
        strategy: str = "mean",
        drop_threshold: float = 0.5,
        knn_neighbors: int = 5,
    ):
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(f"strategy must be one of {_VALID_STRATEGIES}, got '{strategy}'")
        if not 0.0 <= drop_threshold <= 1.0:
            raise ValueError(f"drop_threshold must be in [0, 1], got {drop_threshold}")
        self.strategy = strategy
        self.drop_threshold = drop_threshold
        self.knn_neighbors = knn_neighbors

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Impute missing values and return a cleaned Dataset.

        Steps:
        1. Drop columns where fraction missing > drop_threshold.
        2. Apply strategy to remaining NaN values.

        Parameters
        ----------
        dataset : Dataset

        Returns
        -------
        Dataset
        """
        data = dataset.data.copy()
        feature_names = list(dataset.feature_names)
        feature_types = list(dataset.feature_types)

        # Step 1 — drop high-missing columns
        missing_frac = np.isnan(data).mean(axis=0)
        keep_cols = missing_frac <= self.drop_threshold
        data = data[:, keep_cols]
        feature_names = [n for n, k in zip(feature_names, keep_cols) if k]
        feature_types = [t for t, k in zip(feature_types, keep_cols) if k]

        # Step 2 — impute remaining NaN
        data = self._impute(data)

        return Dataset(
            data=data,
            feature_names=feature_names,
            feature_types=feature_types,
            categorical_data=dict(dataset.categorical_data),
            metadata=dict(dataset.metadata),
        )

    def _impute(self, data: np.ndarray) -> np.ndarray:
        if not np.isnan(data).any():
            return data

        if self.strategy == "mean":
            col_means = np.nanmean(data, axis=0)
            inds = np.where(np.isnan(data))
            data[inds] = col_means[inds[1]]

        elif self.strategy == "median":
            col_medians = np.nanmedian(data, axis=0)
            inds = np.where(np.isnan(data))
            data[inds] = col_medians[inds[1]]

        elif self.strategy == "mode":
            import pandas as pd
            df = pd.DataFrame(data)
            modes = df.mode(axis=0).iloc[0]
            for col_idx in range(data.shape[1]):
                mask = np.isnan(data[:, col_idx])
                if mask.any():
                    data[mask, col_idx] = modes.iloc[col_idx]

        elif self.strategy == "knn":
            from sklearn.impute import KNNImputer
            data = KNNImputer(n_neighbors=self.knn_neighbors).fit_transform(data)

        elif self.strategy == "mice":
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer
            data = IterativeImputer(random_state=0).fit_transform(data)

        elif self.strategy == "drop":
            keep_rows = ~np.isnan(data).any(axis=1)
            data = data[keep_rows]

        return data
