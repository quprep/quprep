"""Outlier detection and handling."""

from __future__ import annotations


class OutlierHandler:
    """
    Detect and handle outliers.

    Parameters
    ----------
    method : str
        Detection method: 'iqr', 'zscore', 'isolation_forest'.
    action : str
        What to do with outliers: 'clip' or 'remove'.
    threshold : float
        IQR multiplier (for 'iqr') or Z-score cutoff (for 'zscore').
        Default 3.0 for Z-score, 1.5 for IQR.
    """

    def __init__(self, method: str = "iqr", action: str = "clip", threshold: float | None = None):
        self.method = method
        self.action = action
        self.threshold = threshold

    def fit_transform(self, dataset):
        """Detect and handle outliers, return cleaned Dataset."""
        raise NotImplementedError("OutlierHandler.fit_transform() — coming in v0.1.0")
