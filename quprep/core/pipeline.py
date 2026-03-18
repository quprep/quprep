"""Pipeline orchestrator — chains ingestion, cleaning, reduction, encoding, and export."""

from __future__ import annotations


class Pipeline:
    """
    Composable preprocessing pipeline for quantum data preparation.

    Each stage is optional and works independently. You can use just the encoder,
    just the reducer, or any combination without touching the rest.

    Parameters
    ----------
    ingester : optional
        Data ingestion component. Auto-detected from source type if omitted.
    cleaner : optional
        Data cleaning component (imputation, outlier handling, categoricals).
    reducer : optional
        Dimensionality reduction component (PCA, LDA, UMAP, etc.).
    normalizer : optional
        Normalization component. Auto-selected per encoding if omitted.
    encoder : required
        Quantum encoding component.
    exporter : optional
        Framework export component. Returns the raw encoded array if omitted.

    Examples
    --------
    >>> pipeline = Pipeline(
    ...     encoder=AngleEncoder(),
    ...     exporter=QiskitExporter(),
    ... )
    >>> result = pipeline.fit_transform(df)
    >>> result.circuit.draw()
    """

    def __init__(
        self,
        ingester=None,
        cleaner=None,
        reducer=None,
        normalizer=None,
        encoder=None,
        exporter=None,
    ):
        self.ingester = ingester
        self.cleaner = cleaner
        self.reducer = reducer
        self.normalizer = normalizer
        self.encoder = encoder
        self.exporter = exporter

    def fit_transform(self, source):
        """Run the full pipeline on source data."""
        raise NotImplementedError("Pipeline.fit_transform() — coming in v0.1.0")

    def fit(self, source):
        """Fit the pipeline on training data."""
        raise NotImplementedError("Pipeline.fit() — coming in v0.1.0")

    def transform(self, source):
        """Transform data using a fitted pipeline."""
        raise NotImplementedError("Pipeline.transform() — coming in v0.1.0")
