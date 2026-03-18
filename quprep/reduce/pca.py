"""Principal Component Analysis reducer."""

from __future__ import annotations


class PCAReducer:
    """
    Reduce dimensionality with PCA.

    A good default for unsupervised tasks. Preserves global variance.
    For classification tasks, consider LDAReducer which preserves class separability
    and has been shown to outperform PCA for QML (Mancilla & Pere, 2022).

    Parameters
    ----------
    n_components : int or float
        Number of components to keep, or variance fraction (e.g. 0.95).
    """

    def __init__(self, n_components: int | float = 0.95):
        self.n_components = n_components

    def fit_transform(self, dataset):
        """Fit PCA and return reduced Dataset."""
        raise NotImplementedError("PCAReducer.fit_transform() — coming in v0.1.0")
