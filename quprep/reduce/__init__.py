"""Dimensionality reduction — PCA, LDA, t-SNE, UMAP, DFT/spectral, hardware-aware."""

from quprep.reduce.hardware_aware import HardwareAwareReducer
from quprep.reduce.lda import LDAReducer
from quprep.reduce.pca import PCAReducer
from quprep.reduce.spectral import SpectralReducer, TSNEReducer, UMAPReducer

__all__ = [
    "PCAReducer",
    "LDAReducer",
    "SpectralReducer",
    "TSNEReducer",
    "UMAPReducer",
    "HardwareAwareReducer",
]
