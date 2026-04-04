"""DFT/spectral and other non-linear reducers (t-SNE, UMAP)."""

from __future__ import annotations

import numpy as np


class SpectralReducer:
    """
    Reduce dimensionality using DFT-based spectral methods.

    Applies a real-valued FFT to each sample (treating features as a
    1-D signal), then retains the n_components lowest-frequency components
    by magnitude. Most noise-robust reduction for time-series and sensor
    data; frequency structure complements amplitude encoding well.

    Parameters
    ----------
    n_components : int
        Number of frequency components to retain. Capped at the number
        of FFT bins (floor(n_features / 2) + 1).
    """

    def __init__(self, n_components: int = 8):
        self.n_components = n_components
        self._fitted = False

    def fit(self, dataset) -> SpectralReducer:
        """
        No-op — SpectralReducer is stateless (FFT has no learned parameters).

        Parameters
        ----------
        dataset : Dataset

        Returns
        -------
        SpectralReducer
        """
        self._fitted = True
        return self

    def transform(self, dataset) -> object:
        """
        Apply row-wise FFT and return the reduced Dataset.

        Parameters
        ----------
        dataset : Dataset

        Returns
        -------
        Dataset
        """
        from quprep.core.dataset import Dataset

        fft = np.fft.rfft(dataset.data, axis=1)
        k = min(self.n_components, fft.shape[1])
        reduced = np.abs(fft[:, :k]).astype(np.float64)
        return Dataset(
            data=reduced,
            feature_names=[f"freq{i}" for i in range(k)],
            feature_types=["continuous"] * k,
            metadata={
                **dataset.metadata,
                "reducer": "spectral",
                "n_components": k,
            },
            categorical_data=dataset.categorical_data,
            labels=dataset.labels,
        )

    def fit_transform(self, dataset):
        """
        Apply row-wise FFT and return the reduced Dataset.

        Parameters
        ----------
        dataset : Dataset
            Input dataset treated as 1-D signal rows.

        Returns
        -------
        Dataset
            Reduced dataset with features named ``freq0``, ``freq1``, etc.
            Values are FFT magnitudes (always >= 0).
        """
        return self.fit(dataset).transform(dataset)


class TSNEReducer:
    """
    t-SNE reducer. Preserves local structure.

    Best for 2–3 qubit visualisation. Note: t-SNE has no transform()
    method — it cannot generalise to new data points.

    Parameters
    ----------
    n_components : int
        Target dimensionality (2 or 3 recommended).
    perplexity : float
        Balances local vs global structure. Typical range: 5–50.
    random_state : int, optional
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        random_state: int | None = 42,
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.random_state = random_state
        self._fitted = False

    def fit(self, dataset) -> TSNEReducer:
        """
        No-op — t-SNE has no generalizable fit (it cannot transform new data).

        Parameters
        ----------
        dataset : Dataset

        Returns
        -------
        TSNEReducer
        """
        self._fitted = True
        return self

    def transform(self, dataset) -> object:
        """
        Re-run t-SNE on the provided dataset.

        .. note::
            t-SNE does not support out-of-sample extension. Calling
            ``transform()`` re-fits from scratch on each call.

        Parameters
        ----------
        dataset : Dataset

        Returns
        -------
        Dataset
        """
        from sklearn.manifold import TSNE

        from quprep.core.dataset import Dataset

        tsne = TSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            random_state=self.random_state,
        )
        reduced = tsne.fit_transform(dataset.data).astype(np.float64)
        k = reduced.shape[1]
        return Dataset(
            data=reduced,
            feature_names=[f"tsne{i}" for i in range(k)],
            feature_types=["continuous"] * k,
            metadata={
                **dataset.metadata,
                "reducer": "tsne",
                "n_components": k,
            },
            categorical_data=dataset.categorical_data,
            labels=dataset.labels,
        )

    def fit_transform(self, dataset):
        """
        Fit t-SNE and return the reduced Dataset.

        Parameters
        ----------
        dataset : Dataset
            Input dataset.

        Returns
        -------
        Dataset
            Reduced dataset with features named ``tsne0``, ``tsne1``, etc.
        """
        return self.fit(dataset).transform(dataset)


class UMAPReducer:
    """
    UMAP reducer. Fast, scalable, preserves local and global structure.

    Requires ``pip install quprep[umap]`` (umap-learn package).

    Parameters
    ----------
    n_components : int
        Target dimensionality.
    n_neighbors : int
        Controls local vs global structure balance.
    random_state : int, optional
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        random_state: int | None = 42,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self._fitted = False
        self._umap = None

    def fit(self, dataset) -> UMAPReducer:
        """
        Fit UMAP on dataset.

        Parameters
        ----------
        dataset : Dataset

        Returns
        -------
        UMAPReducer

        Raises
        ------
        ImportError
            If ``umap-learn`` is not installed.
        """
        try:
            import umap
        except ImportError as e:
            raise ImportError(
                "UMAPReducer requires the umap-learn package. "
                "Install it with: pip install quprep[umap]"
            ) from e
        self._umap = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
        )
        self._umap.fit(dataset.data)
        self._fitted = True
        return self

    def transform(self, dataset) -> object:
        """
        Apply fitted UMAP and return the reduced Dataset.

        Parameters
        ----------
        dataset : Dataset

        Returns
        -------
        Dataset

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
        reduced = self._umap.transform(dataset.data).astype(np.float64)
        k = reduced.shape[1]
        return Dataset(
            data=reduced,
            feature_names=[f"umap{i}" for i in range(k)],
            feature_types=["continuous"] * k,
            metadata={
                **dataset.metadata,
                "reducer": "umap",
                "n_components": k,
            },
            categorical_data=dataset.categorical_data,
            labels=dataset.labels,
        )

    def fit_transform(self, dataset):
        """
        Fit UMAP and return the reduced Dataset.

        Parameters
        ----------
        dataset : Dataset
            Input dataset.

        Returns
        -------
        Dataset
            Reduced dataset with features named ``umap0``, ``umap1``, etc.

        Raises
        ------
        ImportError
            If ``umap-learn`` is not installed (``pip install quprep[umap]``).
        """
        return self.fit(dataset).transform(dataset)
