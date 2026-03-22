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
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        from quprep.core.dataset import Dataset

        _labels = labels if labels is not None else self.labels
        if _labels is None:
            raise ValueError(
                "LDAReducer requires class labels. "
                "Pass labels= to fit_transform() or at LDAReducer(labels=y)."
            )

        _labels = np.asarray(_labels)
        n_classes = len(np.unique(_labels))
        n_features = dataset.data.shape[1]
        # sklearn cap: n_components <= min(n_classes - 1, n_features)
        n = min(self.n_components, n_classes - 1, n_features)

        self._lda = LinearDiscriminantAnalysis(n_components=n)
        reduced = self._lda.fit_transform(dataset.data, _labels)
        k = reduced.shape[1]

        return Dataset(
            data=reduced.astype(np.float64),
            feature_names=[f"ld{i}" for i in range(k)],
            feature_types=["continuous"] * k,
            metadata={
                **dataset.metadata,
                "reducer": "lda",
                "n_components": k,
                "classes": np.unique(_labels).tolist(),
            },
            categorical_data=dataset.categorical_data,
        )
