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
        """
        First item in the batch — convenience for single-sample use.

        Returns the first exported circuit if an exporter was configured,
        otherwise the first ``EncodedResult`` if only an encoder was configured,
        otherwise ``None``.
        """
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

    sklearn-compatible: supports ``fit()``, ``transform()``, ``get_params()``,
    and ``set_params()`` in addition to the native ``fit_transform()``.

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
    schema : DataSchema, optional
        Input schema to validate at pipeline entry. Raises SchemaViolationError
        on mismatch.

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
        schema=None,
    ):
        self.ingester = ingester
        self.cleaner = cleaner
        self.reducer = reducer
        self.normalizer = normalizer
        self.encoder = encoder
        self.exporter = exporter
        self.schema = schema
        self._fitted = False
        self._resolved_normalizer = None

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def fit(self, source, y=None) -> Pipeline:
        """
        Fit all pipeline stages on training data.

        Parameters
        ----------
        source : str, Path, np.ndarray, pd.DataFrame, or Dataset
            Training data.
        y : ignored
            Accepted for sklearn API compatibility.

        Returns
        -------
        Pipeline
            Returns ``self`` for chaining (sklearn convention).
        """
        dataset = self._ingest(source)
        self._validate_entry(dataset)
        self._fit_stages(dataset)
        self._fitted = True
        return self

    def transform(self, source) -> PipelineResult:
        """
        Apply fitted pipeline stages to data.

        Parameters
        ----------
        source : str, Path, np.ndarray, pd.DataFrame, or Dataset
            Input data.

        Returns
        -------
        PipelineResult

        Raises
        ------
        RuntimeError
            If the pipeline has not been fitted yet.
        """
        if not self._fitted:
            raise RuntimeError(
                "Pipeline has not been fitted. Call fit() or fit_transform() first."
            )
        dataset = self._ingest(source)
        return self._apply_stages(dataset)

    def fit_transform(self, source, y=None) -> PipelineResult:
        """
        Fit all stages and transform in a single pass.

        Parameters
        ----------
        source : str, Path, np.ndarray, pd.DataFrame, or Dataset
            Input data.
        y : ignored
            Accepted for sklearn API compatibility.

        Returns
        -------
        PipelineResult
            Contains ``dataset`` (processed), ``encoded`` (list of EncodedResult
            or None), and ``circuits`` (framework-specific circuit objects or None).
        """
        dataset = self._ingest(source)
        self._validate_entry(dataset)
        dataset = self._fit_stages(dataset)
        self._fitted = True
        return self._encode_export(dataset)

    # ------------------------------------------------------------------
    # sklearn estimator interface
    # ------------------------------------------------------------------

    def get_params(self, deep: bool = True) -> dict:
        """
        Return pipeline parameters (sklearn convention).

        Parameters
        ----------
        deep : bool
            Ignored — included for sklearn API compatibility.

        Returns
        -------
        dict
        """
        return {
            "ingester": self.ingester,
            "cleaner": self.cleaner,
            "reducer": self.reducer,
            "normalizer": self.normalizer,
            "encoder": self.encoder,
            "exporter": self.exporter,
            "schema": self.schema,
        }

    def set_params(self, **params) -> Pipeline:
        """
        Set pipeline parameters (sklearn convention).

        Parameters
        ----------
        **params
            Parameter names and values.

        Returns
        -------
        Pipeline
            Returns ``self``.

        Raises
        ------
        ValueError
            If an unknown parameter name is given.
        """
        valid = set(self.get_params())
        for key, value in params.items():
            if key not in valid:
                raise ValueError(
                    f"Invalid parameter '{key}'. Valid parameters: {sorted(valid)}"
                )
            setattr(self, key, value)
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_entry(self, dataset) -> None:
        """Run validation and schema checks at pipeline entry."""
        from quprep.validation.input_validator import validate_dataset
        validate_dataset(dataset, context="at pipeline entry")
        if self.schema is not None:
            self.schema.validate(dataset)

    def _fit_stages(self, dataset):
        """
        Fit all stateful stages sequentially.

        Each stage is fitted on the output of the previous stage, then
        applied to produce the dataset for the next stage.

        Returns the fully transformed dataset (after all fitted stages).
        """
        if self.cleaner is not None:
            self.cleaner.fit(dataset)
            dataset = self.cleaner.transform(dataset)

        if self.reducer is not None:
            self.reducer.fit(dataset)
            dataset = self.reducer.transform(dataset)

        self._resolved_normalizer = self.normalizer
        if self._resolved_normalizer is None and self.encoder is not None:
            key = self._encoding_key()
            if key is not None:
                from quprep.normalize.scalers import auto_normalizer
                self._resolved_normalizer = auto_normalizer(key)

        if self._resolved_normalizer is not None:
            self._resolved_normalizer.fit(dataset)
            dataset = self._resolved_normalizer.transform(dataset)

        # Cost / qubit warning after all reductions are applied
        if self.encoder is not None:
            import warnings

            from quprep.validation.cost import estimate_cost
            from quprep.validation.input_validator import QuPrepWarning
            cost = estimate_cost(self.encoder, dataset.n_features)
            if cost.warning:
                warnings.warn(cost.warning, QuPrepWarning, stacklevel=4)

        return dataset

    def _apply_stages(self, dataset) -> PipelineResult:
        """Apply fitted stages to dataset and return PipelineResult."""
        if self.cleaner is not None:
            dataset = self.cleaner.transform(dataset)

        if self.reducer is not None:
            dataset = self.reducer.transform(dataset)

        if self._resolved_normalizer is not None:
            dataset = self._resolved_normalizer.transform(dataset)

        return self._encode_export(dataset)

    def _encode_export(self, dataset) -> PipelineResult:
        """Run encoder + exporter on a already-transformed dataset."""
        if self.encoder is None:
            return PipelineResult(dataset=dataset, encoded=None, circuits=None)

        encoded = self.encoder.encode_batch(dataset)

        if self.exporter is None:
            return PipelineResult(dataset=dataset, encoded=encoded, circuits=None)

        circuits = self.exporter.export_batch(encoded)
        return PipelineResult(dataset=dataset, encoded=encoded, circuits=circuits)

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
            return "angle_ry"
        if isinstance(self.encoder, HamiltonianEncoder):
            return "hamiltonian"
        return None
