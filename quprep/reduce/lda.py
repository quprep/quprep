"""Linear Discriminant Analysis reducer."""

from __future__ import annotations


class LDAReducer:
    """
    Reduce dimensionality with LDA.

    Preserves class separability. Research shows LDA outperforms PCA for
    quantum classification tasks (Mancilla & Pere, 2022).

    Requires class labels. Maximum components = n_classes - 1.

    Parameters
    ----------
    n_components : int
        Number of discriminant components to keep.
    """

    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def fit_transform(self, dataset, labels=None):
        """Fit LDA and return reduced Dataset."""
        raise NotImplementedError("LDAReducer.fit_transform() — coming in v0.1.0")
