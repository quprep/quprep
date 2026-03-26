"""Principal Component Analysis reducer."""

from __future__ import annotations

import numpy as np


class PCAReducer:
    """
    Reduce dimensionality with PCA.

    A good default for unsupervised tasks. Preserves global variance.
    For classification tasks, consider LDAReducer which preserves class
    separability and has been shown to outperform PCA for QML.

    Parameters
    ----------
    n_components : int or float
        Number of components to keep, or variance fraction (e.g. 0.95 keeps
        95% of variance). Capped at n_features automatically.

    References
    ----------
    Mancilla, J., & Pere, C. (2022). A preprocessing perspective for quantum
        machine learning classification advantage in finance using NISQ algorithms.
        *Entropy*, 24(11), 1656. [doi:10.3390/e24111656](https://doi.org/10.3390/e24111656){target="_blank"}
    """

    def __init__(self, n_components: int | float = 0.95):
        self.n_components = n_components
        self._pca = None
        self._fitted = False

    def fit(self, dataset) -> PCAReducer:
        """
        Fit PCA on dataset.

        Parameters
        ----------
        dataset : Dataset

        Returns
        -------
        PCAReducer
            Returns ``self`` for chaining.
        """
        from sklearn.decomposition import PCA

        n_features = dataset.data.shape[1]
        n = self.n_components
        if isinstance(n, int):
            n = min(n, n_features)

        self._pca = PCA(n_components=n)
        self._pca.fit(dataset.data)
        self._fitted = True
        return self

    def transform(self, dataset) -> object:
        """
        Apply fitted PCA and return a reduced Dataset.

        Parameters
        ----------
        dataset : Dataset

        Returns
        -------
        Dataset
            Reduced dataset with features named ``pc0``, ``pc1``, etc.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If ``fit()`` has not been called yet.
        """
        from sklearn.exceptions import NotFittedError

        from quprep.core.dataset import Dataset

        if not self._fitted:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit()' before 'transform()'."
            )

        reduced = self._pca.transform(dataset.data)
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

    def fit_transform(self, dataset):
        """
        Fit PCA on dataset and return dimensionality-reduced Dataset.

        Parameters
        ----------
        dataset : Dataset
            Input dataset. All features must be numeric.

        Returns
        -------
        Dataset
            Reduced dataset with features named ``pc0``, ``pc1``, etc.
            Explained variance is stored in ``dataset.metadata['explained_variance_ratio']``.
        """
        return self.fit(dataset).transform(dataset)

    @property
    def explained_variance_ratio_(self):
        """
        Variance fraction explained by each component.

        Returns
        -------
        np.ndarray
            Array of length n_components, values sum to <= 1.0.

        Raises
        ------
        RuntimeError
            If ``fit_transform()`` has not been called yet.
        """
        if self._pca is None:
            raise RuntimeError("Call fit() or fit_transform() first.")
        return self._pca.explained_variance_ratio_
