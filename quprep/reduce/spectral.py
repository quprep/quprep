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

    def fit_transform(self, dataset):
        """Apply FFT row-wise and return reduced Dataset."""
        from quprep.core.dataset import Dataset

        fft = np.fft.rfft(dataset.data, axis=1)  # (n_samples, n_freq_bins)
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
        )


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

    def fit_transform(self, dataset):
        """Fit t-SNE and return reduced Dataset."""
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
        )


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

    def fit_transform(self, dataset):
        """Fit UMAP and return reduced Dataset."""
        try:
            import umap
        except ImportError as e:
            raise ImportError(
                "UMAPReducer requires the umap-learn package. "
                "Install it with: pip install quprep[umap]"
            ) from e

        from quprep.core.dataset import Dataset

        reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
        )
        reduced = reducer.fit_transform(dataset.data).astype(np.float64)
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
        )
