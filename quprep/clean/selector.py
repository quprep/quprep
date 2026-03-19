"""Feature selection — correlation, mutual information, variance thresholds."""

from __future__ import annotations

import numpy as np

from quprep.core.dataset import Dataset

_VALID_METHODS = ("correlation", "mutual_info", "variance")


class FeatureSelector:
    """
    Drop redundant or low-signal features before encoding.

    Parameters
    ----------
    method : str
        'correlation' — drop one of any pair of features with |r| > threshold.
                        Keeps the first feature of each correlated pair.
        'mutual_info' — rank features by mutual information with a target;
                        keeps the top `max_features` features.
        'variance'    — drop features with variance below threshold.
    threshold : float
        Correlation cutoff (default 0.95), MI minimum, or variance minimum.
    max_features : int or None
        Hard cap on features kept. Applied after threshold filtering.
        Useful for enforcing a qubit budget. None means no cap.
    """

    def __init__(
        self,
        method: str = "correlation",
        threshold: float = 0.95,
        max_features: int | None = None,
    ):
        if method not in _VALID_METHODS:
            raise ValueError(f"method must be one of {_VALID_METHODS}, got '{method}'")
        self.method = method
        self.threshold = threshold
        self.max_features = max_features

    def fit_transform(self, dataset: Dataset, labels: np.ndarray | None = None) -> Dataset:
        """
        Select features and return a reduced Dataset.

        Parameters
        ----------
        dataset : Dataset
        labels : np.ndarray, optional
            Required for 'mutual_info' method.

        Returns
        -------
        Dataset
        """
        if self.method == "correlation":
            keep = self._correlation_mask(dataset.data)
        elif self.method == "mutual_info":
            keep = self._mutual_info_mask(dataset.data, labels)
        else:
            keep = self._variance_mask(dataset.data)

        if self.max_features is not None:
            # among kept features, apply hard cap (left-to-right)
            kept_indices = np.where(keep)[0][: self.max_features]
            keep = np.zeros(dataset.n_features, dtype=bool)
            keep[kept_indices] = True

        data = dataset.data[:, keep]
        feature_names = [n for n, k in zip(dataset.feature_names, keep) if k]
        feature_types = [t for t, k in zip(dataset.feature_types, keep) if k]

        return Dataset(
            data=data,
            feature_names=feature_names,
            feature_types=feature_types,
            categorical_data=dict(dataset.categorical_data),
            metadata=dict(dataset.metadata),
        )

    def _correlation_mask(self, data: np.ndarray) -> np.ndarray:
        n = data.shape[1]
        keep = np.ones(n, dtype=bool)
        # use nanmean-based correlation to handle NaN
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(data.T)
        if corr.ndim == 0:
            return keep
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(i + 1, n):
                if keep[j] and abs(corr[i, j]) > self.threshold:
                    keep[j] = False
        return keep

    def _mutual_info_mask(self, data: np.ndarray, labels) -> np.ndarray:
        if labels is None:
            raise ValueError("'mutual_info' method requires labels.")
        from sklearn.feature_selection import mutual_info_classif
        mi = mutual_info_classif(data, labels, random_state=0)
        keep = mi >= self.threshold
        if not keep.any():
            # fallback: keep at least top-1
            keep[np.argmax(mi)] = True
        return keep

    def _variance_mask(self, data: np.ndarray) -> np.ndarray:
        variances = np.nanvar(data, axis=0)
        return variances >= self.threshold
