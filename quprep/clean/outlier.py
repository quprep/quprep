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
        data = dataset.data.copy()

        if self.method == "iqr":
            data, keep_rows = self._iqr(data)
        elif self.method == "zscore":
            data, keep_rows = self._zscore(data)
        else:
            data, keep_rows = self._isolation_forest(data)

        cat_data = {}
        for col, vals in dataset.categorical_data.items():
            cat_data[col] = [v for v, k in zip(vals, keep_rows) if k]

        return Dataset(
            data=data,
            feature_names=list(dataset.feature_names),
            feature_types=list(dataset.feature_types),
            categorical_data=cat_data,
            metadata=dict(dataset.metadata),
        )

    def _iqr(self, data: np.ndarray):
        q1 = np.nanpercentile(data, 25, axis=0)
        q3 = np.nanpercentile(data, 75, axis=0)
        iqr = q3 - q1
        lower = q1 - self.threshold * iqr
        upper = q3 + self.threshold * iqr
        return self._apply(data, lower, upper)

    def _zscore(self, data: np.ndarray):
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        std = np.where(std == 0, 1.0, std)
        lower = mean - self.threshold * std
        upper = mean + self.threshold * std
        return self._apply(data, lower, upper)

    def _apply(self, data: np.ndarray, lower: np.ndarray, upper: np.ndarray):
        n = data.shape[0]
        if self.action == "clip":
            data = np.clip(data, lower, upper)
            return data, np.ones(n, dtype=bool)
        else:
            is_outlier = (data < lower) | (data > upper)
            keep = ~is_outlier.any(axis=1)
            return data[keep], keep

    def _isolation_forest(self, data: np.ndarray):
        from sklearn.ensemble import IsolationForest
        clean = data[~np.isnan(data).any(axis=1)]
        clf = IsolationForest(contamination=self.contamination, random_state=0)
        preds = clf.fit_predict(clean)

        n = data.shape[0]
        # map predictions back (NaN rows are treated as inliers)
        full_preds = np.ones(n, dtype=int)
        clean_idx = np.where(~np.isnan(data).any(axis=1))[0]
        for i, idx in enumerate(clean_idx):
            full_preds[idx] = preds[i]

        inlier_mask = full_preds == 1

        if self.action == "remove":
            return data[inlier_mask], inlier_mask

        # clip: clamp to inlier min/max per feature
        inlier_data = data[inlier_mask]
        lower = np.nanmin(inlier_data, axis=0)
        upper = np.nanmax(inlier_data, axis=0)
        data = np.clip(data, lower, upper)
        return data, np.ones(n, dtype=bool)
