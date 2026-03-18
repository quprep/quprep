"""Feature selection — correlation, mutual information, variance thresholds."""

from __future__ import annotations


class FeatureSelector:
    """
    Select the most informative features.

    Parameters
    ----------
    method : str
        'correlation', 'mutual_info', or 'variance'.
    threshold : float
        Cutoff value (correlation coefficient, MI score, or variance).
    max_features : int, optional
        Keep at most this many features. Useful for enforcing qubit budgets.
    """

    def __init__(self, method: str = "correlation", threshold: float = 0.95, max_features: int | None = None):
        self.method = method
        self.threshold = threshold
        self.max_features = max_features

    def fit_transform(self, dataset):
        """Select features and return reduced Dataset."""
        raise NotImplementedError("FeatureSelector.fit_transform() — coming in v0.1.0")
