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
        self._fitted = False
        self._keep_cols: np.ndarray | None = None
        self._fill_values: np.ndarray | None = None  # for mean/median/mode
        self._sklearn_imputer = None                  # for knn/mice

    def fit(self, dataset: Dataset) -> Imputer:
        """
        Learn imputation parameters from dataset.

        Parameters
        ----------
        dataset : Dataset

        Returns
        -------
        Imputer
            Returns ``self`` for chaining.
        """
        data = dataset.data
        missing_frac = np.isnan(data).mean(axis=0)
        self._keep_cols = missing_frac <= self.drop_threshold
        kept = data[:, self._keep_cols]

        if self.strategy == "mean":
            self._fill_values = np.nanmean(kept, axis=0)
        elif self.strategy == "median":
            self._fill_values = np.nanmedian(kept, axis=0)
        elif self.strategy == "mode":
            import pandas as pd
            df = pd.DataFrame(kept)
            self._fill_values = df.mode(axis=0).iloc[0].to_numpy(dtype=float)
        elif self.strategy == "knn":
            from sklearn.impute import KNNImputer
            self._sklearn_imputer = KNNImputer(n_neighbors=self.knn_neighbors)
            self._sklearn_imputer.fit(kept)
        elif self.strategy == "mice":
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer
            self._sklearn_imputer = IterativeImputer(random_state=0)
            self._sklearn_imputer.fit(kept)
        # 'drop' is stateless — nothing to learn

        self._fitted = True
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Apply learned imputation and return a cleaned Dataset.

        For ``strategy='drop'``, rows containing NaN are removed at transform
        time (this cannot be learned from training data as test rows differ).

        Parameters
        ----------
        dataset : Dataset

        Returns
        -------
        Dataset

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If ``fit()`` has not been called yet.
        """
        from sklearn.exceptions import NotFittedError

        if not self._fitted:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit()' before 'transform()'."
            )

        data = dataset.data.copy()
        feature_names = list(dataset.feature_names)
        feature_types = list(dataset.feature_types)

        # Apply column drop from fit
        data = data[:, self._keep_cols]
        feature_names = [n for n, k in zip(feature_names, self._keep_cols) if k]
        feature_types = [t for t, k in zip(feature_types, self._keep_cols) if k]

        data = self._apply_imputation(data)

        return Dataset(
            data=data,
            feature_names=feature_names,
            feature_types=feature_types,
            categorical_data=dict(dataset.categorical_data),
            metadata=dict(dataset.metadata),
        )

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
        return self.fit(dataset).transform(dataset)

    def _apply_imputation(self, data: np.ndarray) -> np.ndarray:
        """Apply fitted imputation to data (after column drop has been applied)."""
        if not np.isnan(data).any():
            return data

        if self.strategy == "mean":
            inds = np.where(np.isnan(data))
            data[inds] = self._fill_values[inds[1]]

        elif self.strategy == "median":
            inds = np.where(np.isnan(data))
            data[inds] = self._fill_values[inds[1]]

        elif self.strategy == "mode":
            for col_idx in range(data.shape[1]):
                mask = np.isnan(data[:, col_idx])
                if mask.any():
                    data[mask, col_idx] = self._fill_values[col_idx]

        elif self.strategy in ("knn", "mice"):
            data = self._sklearn_imputer.transform(data)

        elif self.strategy == "drop":
            keep_rows = ~np.isnan(data).any(axis=1)
            data = data[keep_rows]

        return data
