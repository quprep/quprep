"""Pipeline orchestrator — chains ingestion, cleaning, reduction, encoding, and export."""

from __future__ import annotations

from pathlib import Path


class PipelineResult:
    """
    Output of Pipeline.fit_transform().

    Attributes
    ----------
    dataset : Dataset
        The processed Dataset after all pipeline stages (post-normalization).
    encoded : list[EncodedResult] or None
        One EncodedResult per sample. None if no encoder was configured.
    circuits : list or None
        Exported circuit objects (framework-specific). None if no exporter was configured.
    """

    def __init__(self, dataset, encoded, circuits):
        self.dataset = dataset
        self.encoded = encoded
        self.circuits = circuits

    @property
    def circuit(self):
        """First circuit in the batch — convenience for single-sample use."""
        if self.circuits:
            return self.circuits[0]
        if self.encoded:
            return self.encoded[0]
        return None

    def __repr__(self) -> str:
        n = len(self.encoded) if self.encoded else 0
        has_circuits = self.circuits is not None
        return (
            f"PipelineResult(n_samples={n}, "
            f"circuits={'yes' if has_circuits else 'no'})"
        )


class Pipeline:
    """
    Composable preprocessing pipeline for quantum data preparation.

    Each stage is optional and works independently. You can use just the
    encoder, just the reducer, or any combination without touching the rest.

    Parameters
    ----------
    ingester : optional
        Data ingestion component. Auto-detected from source type if omitted.
    cleaner : optional
        Data cleaning component (Imputer, OutlierHandler, CategoricalEncoder).
    reducer : optional
        Dimensionality reduction component (PCA, LDA, etc.).
    normalizer : optional
        Normalization component. Auto-selected per encoding if omitted.
    encoder : optional
        Quantum encoding component. Returns a processed Dataset if omitted.
    exporter : optional
        Framework export component. Returns EncodedResult list if omitted.

    Examples
    --------
    >>> pipeline = Pipeline(
    ...     encoder=AngleEncoder(),
    ...     exporter=QASMExporter(),
    ... )
    >>> result = pipeline.fit_transform(df)
    >>> print(result.circuits[0])
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

    def fit_transform(self, source) -> PipelineResult:
        """Run all pipeline stages on source data."""
        dataset = self._ingest(source)

        if self.cleaner is not None:
            dataset = self.cleaner.fit_transform(dataset)

        if self.reducer is not None:
            dataset = self.reducer.fit_transform(dataset)

        normalizer = self.normalizer
        if normalizer is None and self.encoder is not None:
            key = self._encoding_key()
            if key is not None:
                from quprep.normalize.scalers import auto_normalizer
                normalizer = auto_normalizer(key)
        if normalizer is not None:
            dataset = normalizer.fit_transform(dataset)

        if self.encoder is None:
            return PipelineResult(dataset=dataset, encoded=None, circuits=None)

        encoded = self.encoder.encode_batch(dataset)

        if self.exporter is None:
            return PipelineResult(dataset=dataset, encoded=encoded, circuits=None)

        circuits = self.exporter.export_batch(encoded)
        return PipelineResult(dataset=dataset, encoded=encoded, circuits=circuits)

    def fit(self, source):
        """Fit the pipeline on training data."""
        raise NotImplementedError("Pipeline.fit() — coming in v0.2.0")

    def transform(self, source):
        """Transform data using a fitted pipeline."""
        raise NotImplementedError("Pipeline.transform() — coming in v0.2.0")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ingest(self, source):
        """Return a Dataset regardless of what source type is passed in."""
        from quprep.core.dataset import Dataset

        if isinstance(source, Dataset):
            return source

        if self.ingester is not None:
            return self.ingester.load(source)

        # Auto-detect source type
        import numpy as np

        if isinstance(source, (str, Path)):
            from quprep.ingest.csv_ingester import CSVIngester
            return CSVIngester().load(source)

        if isinstance(source, (np.ndarray, list)):
            from quprep.ingest.numpy_ingester import NumpyIngester
            return NumpyIngester().load(source)

        try:
            import pandas as pd
            if isinstance(source, pd.DataFrame):
                from quprep.ingest.numpy_ingester import NumpyIngester
                return NumpyIngester().load(source)
        except ImportError:
            pass

        raise TypeError(
            f"Cannot ingest source of type '{type(source).__name__}'. "
            "Pass a file path, np.ndarray, pd.DataFrame, or Dataset."
        )

    def _encoding_key(self) -> str | None:
        """Map the configured encoder to its auto_normalizer key."""
        from quprep.encode.amplitude import AmplitudeEncoder
        from quprep.encode.angle import AngleEncoder
        from quprep.encode.basis import BasisEncoder
        from quprep.encode.entangled_angle import EntangledAngleEncoder
        from quprep.encode.hamiltonian import HamiltonianEncoder
        from quprep.encode.iqp import IQPEncoder
        from quprep.encode.reupload import ReUploadEncoder

        if isinstance(self.encoder, (AngleEncoder, EntangledAngleEncoder)):
            return f"angle_{self.encoder.rotation}"
        if isinstance(self.encoder, AmplitudeEncoder):
            return "amplitude"
        if isinstance(self.encoder, BasisEncoder):
            return "basis"
        if isinstance(self.encoder, (IQPEncoder, ReUploadEncoder)):
            return "angle_ry"  # both need [-π, π] like angle Ry
        if isinstance(self.encoder, HamiltonianEncoder):
            return "hamiltonian"
        return None
