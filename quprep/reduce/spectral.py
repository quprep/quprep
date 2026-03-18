"""DFT/spectral and other non-linear reducers (t-SNE, UMAP)."""

from __future__ import annotations


class SpectralReducer:
    """
    Reduce dimensionality using DFT-based spectral methods.

    Most noise-robust reduction method for time-series and signal data
    (2025 research). Retains frequency structure, which is highly
    complementary to quantum amplitude encoding.

    Parameters
    ----------
    n_components : int
        Number of frequency components to retain.
    """

    def __init__(self, n_components: int = 8):
        self.n_components = n_components

    def fit_transform(self, dataset):
        raise NotImplementedError("SpectralReducer.fit_transform() — coming in v0.2.0")


class TSNEReducer:
    """t-SNE reducer. Preserves local structure. Best for 2–3 qubit visualisation."""

    def __init__(self, n_components: int = 2, perplexity: float = 30.0):
        self.n_components = n_components
        self.perplexity = perplexity

    def fit_transform(self, dataset):
        raise NotImplementedError("TSNEReducer.fit_transform() — coming in v0.2.0")


class UMAPReducer:
    """UMAP reducer. Fast, scalable, preserves local and global structure."""

    def __init__(self, n_components: int = 2, n_neighbors: int = 15):
        self.n_components = n_components
        self.n_neighbors = n_neighbors

    def fit_transform(self, dataset):
        raise NotImplementedError("UMAPReducer.fit_transform() — coming in v0.2.0")
