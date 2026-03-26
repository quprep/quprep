"""Outlier detection and handling."""

from __future__ import annotations

import numpy as np

from quprep.core.dataset import Dataset

_VALID_METHODS = ("iqr", "zscore", "isolation_forest")
_VALID_ACTIONS = ("clip", "remove")

_DEFAULT_THRESHOLD = {"iqr": 1.5, "zscore": 3.0, "isolation_forest": None}


class OutlierHandler:
    """
    Detect and handle outliers in a Dataset.

    Parameters
    ----------
    method : str
        'iqr'              — interquartile range rule.
        'zscore'           — standard deviation rule.
        'isolation_forest' — sklearn IsolationForest.
    action : str
        'clip'   — clamp outlier values to the detection boundary.
        'remove' — drop rows that contain at least one outlier.
        For 'isolation_forest', 'clip' clips to the min/max of inlier data.
    threshold : float or None
        IQR multiplier (default 1.5) or Z-score cutoff (default 3.0).
        Ignored for 'isolation_forest'.
    contamination : float
        Expected fraction of outliers. Only used by 'isolation_forest'.
        Default 'auto'.
    """

    def __init__(
        self,
        method: str = "iqr",
        action: str = "clip",
        threshold: float | None = None,
        contamination: float | str = "auto",
    ):
        if method not in _VALID_METHODS:
            raise ValueError(f"method must be one of {_VALID_METHODS}, got '{method}'")
        if action not in _VALID_ACTIONS:
            raise ValueError(f"action must be one of {_VALID_ACTIONS}, got '{action}'")
        self.method = method
        self.action = action
        self.threshold = threshold if threshold is not None else _DEFAULT_THRESHOLD[method]
        self.contamination = contamination
        self._fitted = False
        self._lower: np.ndarray | None = None
        self._upper: np.ndarray | None = None
        self._iso_forest = None

    def fit(self, dataset: Dataset) -> OutlierHandler:
        """
        Learn outlier detection boundaries from dataset.

        Parameters
        ----------
        dataset : Dataset

        Returns
        -------
        OutlierHandler
            Returns ``self`` for chaining.
        """
        data = dataset.data
        if self.method == "iqr":
            q1 = np.nanpercentile(data, 25, axis=0)
            q3 = np.nanpercentile(data, 75, axis=0)
            iqr = q3 - q1
            self._lower = q1 - self.threshold * iqr
            self._upper = q3 + self.threshold * iqr
        elif self.method == "zscore":
            mean = np.nanmean(data, axis=0)
            std = np.nanstd(data, axis=0)
            std = np.where(std == 0, 1.0, std)
            self._lower = mean - self.threshold * std
            self._upper = mean + self.threshold * std
        else:  # isolation_forest
            from sklearn.ensemble import IsolationForest
            clean = data[~np.isnan(data).any(axis=1)]
            self._iso_forest = IsolationForest(
                contamination=self.contamination, random_state=0
            )
            self._iso_forest.fit(clean)
            # store inlier bounds for clip action
            preds = self._iso_forest.predict(clean)
            inlier_data = clean[preds == 1]
            self._lower = np.nanmin(inlier_data, axis=0)
            self._upper = np.nanmax(inlier_data, axis=0)
        self._fitted = True
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Apply outlier handling and return a cleaned Dataset.

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

        if self.method == "isolation_forest":
            data, keep_rows = self._apply_iso(data)
        else:
            data, keep_rows = self._apply(data, self._lower, self._upper)

        cat_data = {
            col: [v for v, k in zip(vals, keep_rows) if k]
            for col, vals in dataset.categorical_data.items()
        }
        return Dataset(
            data=data,
            feature_names=list(dataset.feature_names),
            feature_types=list(dataset.feature_types),
            categorical_data=cat_data,
            metadata=dict(dataset.metadata),
        )

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Detect and handle outliers, return cleaned Dataset.

        Parameters
        ----------
        dataset : Dataset

        Returns
        -------
        Dataset
        """
        return self.fit(dataset).transform(dataset)

    def _apply(self, data: np.ndarray, lower: np.ndarray, upper: np.ndarray):
        n = data.shape[0]
        if self.action == "clip":
            data = np.clip(data, lower, upper)
            return data, np.ones(n, dtype=bool)
        else:
            is_outlier = (data < lower) | (data > upper)
            keep = ~is_outlier.any(axis=1)
            return data[keep], keep

    def _apply_iso(self, data: np.ndarray):
        n = data.shape[0]
        full_preds = np.ones(n, dtype=int)
        clean_idx = np.where(~np.isnan(data).any(axis=1))[0]
        if clean_idx.size:
            preds = self._iso_forest.predict(data[clean_idx])
            for i, idx in enumerate(clean_idx):
                full_preds[idx] = preds[i]
        inlier_mask = full_preds == 1
        if self.action == "remove":
            return data[inlier_mask], inlier_mask
        data = np.clip(data, self._lower, self._upper)
        return data, np.ones(n, dtype=bool)
