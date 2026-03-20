"""Principal Component Analysis reducer."""

from __future__ import annotations

import numpy as np


class PCAReducer:
    """
    Reduce dimensionality with PCA.

    A good default for unsupervised tasks. Preserves global variance.
    For classification tasks, consider LDAReducer which preserves class
    separability and has been shown to outperform PCA for QML
    (Mancilla & Pere, 2022).

    Parameters
    ----------
    n_components : int or float
        Number of components to keep, or variance fraction (e.g. 0.95 keeps
        95% of variance). Capped at n_features automatically.
    """

    def __init__(self, n_components: int | float = 0.95):
        self.n_components = n_components
        self._pca = None

    def fit_transform(self, dataset):
        """Fit PCA and return reduced Dataset."""
        from sklearn.decomposition import PCA

        from quprep.core.dataset import Dataset

        n_features = dataset.data.shape[1]
        n = self.n_components
        if isinstance(n, int):
            n = min(n, n_features)

        self._pca = PCA(n_components=n)
        reduced = self._pca.fit_transform(dataset.data)
        k = reduced.shape[1]

        return Dataset(
            data=reduced.astype(np.float64),
            feature_names=[f"pc{i}" for i in range(k)],
            feature_types=["continuous"] * k,
            metadata={
                **dataset.metadata,
                "reducer": "pca",
                "n_components": k,
                "explained_variance_ratio": self._pca.explained_variance_ratio_.tolist(),
            },
            categorical_data=dataset.categorical_data,
        )

    @property
    def explained_variance_ratio_(self):
        """Variance fraction explained by each component (after fit)."""
        if self._pca is None:
            raise RuntimeError("Call fit_transform() first.")
        return self._pca.explained_variance_ratio_
