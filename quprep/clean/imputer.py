"""Missing value imputation strategies."""

from __future__ import annotations


class Imputer:
    """
    Handle missing values in a Dataset.

    Parameters
    ----------
    strategy : str
        'mean', 'median', 'mode', 'knn', 'mice', or 'drop'.
        'knn' uses k-nearest neighbours imputation.
        'mice' uses iterative (chained equations) imputation.
    drop_threshold : float
        Drop a feature entirely if more than this fraction of values are missing.
        Default 0.5.
    """

    def __init__(self, strategy: str = "mean", drop_threshold: float = 0.5):
        self.strategy = strategy
        self.drop_threshold = drop_threshold

    def fit_transform(self, dataset):
        """Fit imputer and return cleaned Dataset."""
        raise NotImplementedError("Imputer.fit_transform() — coming in v0.1.0")
