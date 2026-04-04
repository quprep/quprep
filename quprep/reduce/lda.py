"""Linear Discriminant Analysis reducer."""

from __future__ import annotations

import numpy as np


class LDAReducer:
    """
    Reduce dimensionality with LDA.

    Preserves class separability. Research shows LDA outperforms PCA for
    quantum classification tasks.

    Requires class labels — pass at init or directly to fit_transform().
    Maximum components = n_classes - 1 (sklearn enforces this automatically).

    Parameters
    ----------
    n_components : int
        Number of discriminant components to keep.
    labels : array-like, optional
        Class labels. Can also be passed directly to fit_transform().

    References
    ----------
    Mancilla, J., & Pere, C. (2022). A preprocessing perspective for quantum
        machine learning classification advantage in finance using NISQ algorithms.
        *Entropy*, 24(11), 1656. [doi:10.3390/e24111656](https://doi.org/10.3390/e24111656){target="_blank"}
    """

    def __init__(self, n_components: int = 2, labels=None):
        self.n_components = n_components
        self.labels = labels
        self._lda = None
        self._fitted = False
        self._fitted_labels: np.ndarray | None = None  # stored for metadata

    def fit(self, dataset, labels=None) -> LDAReducer:
        """
        Fit LDA on dataset.

        Parameters
        ----------
        dataset : Dataset
        labels : array-like, optional
            Class labels. Overrides ``self.labels`` if provided.

        Returns
        -------
        LDAReducer
            Returns ``self`` for chaining.

        Raises
        ------
        ValueError
            If no labels are available.
        """
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        _labels = labels if labels is not None else self.labels
        if _labels is None:
            raise ValueError(
                "LDAReducer requires class labels. "
                "Pass labels= to fit() or at LDAReducer(labels=y)."
            )
        _labels = np.asarray(_labels)
        n_classes = len(np.unique(_labels))
        n_features = dataset.data.shape[1]
        n = min(self.n_components, n_classes - 1, n_features)
        self._lda = LinearDiscriminantAnalysis(n_components=n)
        self._lda.fit(dataset.data, _labels)
        self._fitted_labels = _labels
        self._fitted = True
        return self

    def transform(self, dataset) -> object:
        """
        Apply fitted LDA and return a reduced Dataset.

        Parameters
        ----------
        dataset : Dataset

        Returns
        -------
        Dataset
            Reduced dataset with features named ``ld0``, ``ld1``, etc.

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
        reduced = self._lda.transform(dataset.data)
        k = reduced.shape[1]
        return Dataset(
            data=reduced.astype(np.float64),
            feature_names=[f"ld{i}" for i in range(k)],
            feature_types=["continuous"] * k,
            metadata={
                **dataset.metadata,
                "reducer": "lda",
                "n_components": k,
                "classes": np.unique(self._fitted_labels).tolist(),
            },
            categorical_data=dataset.categorical_data,
            labels=dataset.labels,
        )

    def fit_transform(self, dataset, labels=None):
        """
        Fit LDA on dataset and return dimensionality-reduced Dataset.

        Parameters
        ----------
        dataset : Dataset
            Input dataset. All features must be numeric.
        labels : array-like, optional
            Class labels. Overrides ``self.labels`` if provided.
            Required if ``labels`` was not set at init.

        Returns
        -------
        Dataset
            Reduced dataset with features named ``ld0``, ``ld1``, etc.
            Actual n_components is capped at ``n_classes - 1``.

        Raises
        ------
        ValueError
            If no labels are available (neither at init nor passed here).
        """
        return self.fit(dataset, labels).transform(dataset)
