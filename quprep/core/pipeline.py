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
    cost : CostEstimate or None
        Gate-count and NISQ-safety estimate for the chosen encoder. None if no encoder
        was configured.
    audit_log : list[dict] or None
        One entry per preprocessing stage that ran, in order. Each dict has keys:
        ``stage``, ``n_samples_in``, ``n_features_in``, ``n_samples_out``,
        ``n_features_out``. None if no preprocessing stages ran.
    """

    def __init__(self, dataset, encoded, circuits, cost=None, audit_log=None, drift_report=None):
        self.dataset = dataset
        self.encoded = encoded
        self.circuits = circuits
        self.cost = cost
        self.audit_log = audit_log
        self.drift_report = drift_report

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

    def summary(self) -> str:
        """
        Return a human-readable report of the pipeline result.

        Includes the audit log as a formatted table (if any preprocessing
        stages ran) and the cost estimate breakdown (if an encoder was used).

        Returns
        -------
        str
        """
        lines = ["PipelineResult"]

        n_samples = len(self.encoded) if self.encoded else (
            self.dataset.n_samples if self.dataset is not None else 0
        )
        n_features = self.dataset.n_features if self.dataset is not None else 0
        lines.append(f"  samples  : {n_samples}")
        lines.append(f"  features : {n_features} (post-preprocessing)")

        if self.audit_log:
            lines.append("")
            lines.append("  Preprocessing stages:")
            col_w = 12
            header = (
                f"  {'stage':<{col_w}}  {'samples in':>10}  {'feat in':>7}"
                f"  {'samples out':>11}  {'feat out':>8}"
            )
            lines.append(header)
            lines.append("  " + "-" * (len(header) - 2))
            for entry in self.audit_log:
                lines.append(
                    f"  {entry['stage']:<{col_w}}  "
                    f"{entry['n_samples_in']:>10}  "
                    f"{entry['n_features_in']:>7}  "
                    f"{entry['n_samples_out']:>11}  "
                    f"{entry['n_features_out']:>8}"
                )

        if self.cost is not None:
            c = self.cost
            lines.append("")
            lines.append("  Cost estimate:")
            lines.append(f"    encoding     : {c.encoding}")
            lines.append(f"    qubits       : {c.n_qubits}")
            lines.append(f"    gate count   : {c.gate_count}")
            lines.append(f"    circuit depth: {c.circuit_depth}")
            lines.append(f"    2-qubit gates: {c.two_qubit_gates}")
            nisq_label = "yes" if c.nisq_safe else "NO — exceeds NISQ thresholds"
            lines.append(f"    NISQ-safe    : {nisq_label}")
            if c.warning:
                lines.append(f"    warning      : {c.warning}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        n = len(self.encoded) if self.encoded else 0
        has_circuits = self.circuits is not None
        nisq = (
            f", nisq_safe={self.cost.nisq_safe}"
            if self.cost is not None
            else ""
        )
        return (
            f"PipelineResult(n_samples={n}, "
            f"circuits={'yes' if has_circuits else 'no'}"
            f"{nisq})"
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
    preprocessor : optional
        Preprocessing step applied after ingestion. Accepts a single transformer
        or a list of transformers applied in order (e.g. ``[WindowTransformer(), ...]``).
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
        preprocessor=None,
        cleaner=None,
        reducer=None,
        normalizer=None,
        encoder=None,
        exporter=None,
        schema=None,
        drift_detector=None,
    ):
        self.ingester = ingester
        self.preprocessor = preprocessor
        self.cleaner = cleaner
        self.reducer = reducer
        self.normalizer = normalizer
        self.encoder = encoder
        self.exporter = exporter
        self.schema = schema
        self.drift_detector = drift_detector
        self._fitted = False
        self._resolved_normalizer = None
        self._last_cost = None
        self._last_audit_log = None
        self._last_drift_report = None

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
        y : np.ndarray or array-like, optional
            Target labels. Stored in ``Dataset.labels`` and passed to
            ``FeatureSelector`` when using the ``'mutual_info'`` method.
            Ignored if labels are already embedded in the Dataset (e.g. via
            ``CSVIngester(target_columns=...)``).

        Returns
        -------
        Pipeline
            Returns ``self`` for chaining (sklearn convention).
        """
        import numpy as np
        dataset = self._ingest(source)
        if y is not None and dataset.labels is None:
            dataset.labels = np.asarray(y)
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
        y : np.ndarray or array-like, optional
            Target labels. Stored in ``Dataset.labels`` and passed to
            ``FeatureSelector`` when using the ``'mutual_info'`` method.
            Ignored if labels are already embedded in the Dataset.

        Returns
        -------
        PipelineResult
            Contains ``dataset`` (processed), ``encoded`` (list of EncodedResult
            or None), and ``circuits`` (framework-specific circuit objects or None).
        """
        import numpy as np
        dataset = self._ingest(source)
        if y is not None and dataset.labels is None:
            dataset.labels = np.asarray(y)
        self._validate_entry(dataset)
        dataset = self._fit_stages(dataset)
        self._fitted = True
        return self._encode_export(dataset)

    # ------------------------------------------------------------------
    # sklearn estimator interface
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """
        Return a human-readable snapshot of the pipeline configuration.

        Shows which stages are configured, whether the pipeline has been
        fitted, the resolved normalizer, and the last cost estimate (if
        available).

        Returns
        -------
        str
        """
        lines = ["Pipeline"]
        lines.append(f"  fitted       : {'yes' if self._fitted else 'no'}")

        stage_names = [
            ("ingester",     self.ingester),
            ("preprocessor", self.preprocessor),
            ("cleaner",      self.cleaner),
            ("reducer",      self.reducer),
            ("normalizer",   self._resolved_normalizer or self.normalizer),
            ("encoder",      self.encoder),
            ("exporter",     self.exporter),
        ]
        for name, stage in stage_names:
            if stage is not None:
                lines.append(f"  {name:<12} : {type(stage).__name__}")

        if self.schema is not None:
            lines.append(f"  schema       : {len(self.schema.features)} feature(s)")

        if self._last_cost is not None:
            c = self._last_cost
            lines.append(
                f"  cost         : {c.encoding} | "
                f"{c.n_qubits} qubits | "
                f"depth {c.circuit_depth} | "
                f"gates {c.gate_count} | "
                f"NISQ-safe {'yes' if c.nisq_safe else 'NO'}"
            )

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

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
            "preprocessor": self.preprocessor,
            "cleaner": self.cleaner,
            "reducer": self.reducer,
            "normalizer": self.normalizer,
            "encoder": self.encoder,
            "exporter": self.exporter,
            "schema": self.schema,
            "drift_detector": self.drift_detector,
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

    def save(self, path: str | Path) -> None:
        """
        Persist the pipeline (configuration and fitted state) to a file.

        Uses Python's ``pickle`` protocol. The saved file can be reloaded
        with :meth:`Pipeline.load` and applied to new data without re-fitting.

        Parameters
        ----------
        path : str or Path
            Destination file path (e.g. ``'pipeline.pkl'``). Parent
            directories are created automatically.
        """
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> Pipeline:
        """
        Load a previously saved pipeline from a file.

        Parameters
        ----------
        path : str or Path
            Path to a file created by :meth:`Pipeline.save`.

        Returns
        -------
        Pipeline

        Raises
        ------
        TypeError
            If the file does not contain a Pipeline object.
        """
        import pickle

        with open(Path(path), "rb") as f:
            obj = pickle.load(f)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected a Pipeline object, got {type(obj).__name__}."
            )
        return obj

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
        Also populates ``self._last_audit_log`` and ``self._last_cost``.
        """
        audit: list[dict] = []

        if self.preprocessor is not None:
            _preprocessors = (
                self.preprocessor
                if isinstance(self.preprocessor, list)
                else [self.preprocessor]
            )
            for i, pre in enumerate(_preprocessors):
                n_s_in, n_f_in = dataset.n_samples, dataset.n_features
                dataset = pre.fit_transform(dataset)
                label = "preprocessor" if len(_preprocessors) == 1 else f"preprocessor[{i}]"
                audit.append({
                    "stage": label,
                    "n_samples_in": n_s_in, "n_features_in": n_f_in,
                    "n_samples_out": dataset.n_samples, "n_features_out": dataset.n_features,
                })

        if self.cleaner is not None:
            from quprep.clean.selector import FeatureSelector
            n_s_in, n_f_in = dataset.n_samples, dataset.n_features
            if isinstance(self.cleaner, FeatureSelector):
                self.cleaner.fit(dataset, labels=dataset.labels)
            else:
                self.cleaner.fit(dataset)
            dataset = self.cleaner.transform(dataset)
            audit.append({
                "stage": "cleaner",
                "n_samples_in": n_s_in, "n_features_in": n_f_in,
                "n_samples_out": dataset.n_samples, "n_features_out": dataset.n_features,
            })

        if self.reducer is not None:
            n_s_in, n_f_in = dataset.n_samples, dataset.n_features
            self.reducer.fit(dataset)
            dataset = self.reducer.transform(dataset)
            audit.append({
                "stage": "reducer",
                "n_samples_in": n_s_in, "n_features_in": n_f_in,
                "n_samples_out": dataset.n_samples, "n_features_out": dataset.n_features,
            })

        self._resolved_normalizer = self.normalizer
        if self._resolved_normalizer is None and self.encoder is not None:
            key = self._encoding_key()
            if key is not None:
                from quprep.normalize.scalers import auto_normalizer
                self._resolved_normalizer = auto_normalizer(key)

        if self._resolved_normalizer is not None:
            n_s_in, n_f_in = dataset.n_samples, dataset.n_features
            self._resolved_normalizer.fit(dataset)
            dataset = self._resolved_normalizer.transform(dataset)
            audit.append({
                "stage": "normalizer",
                "n_samples_in": n_s_in, "n_features_in": n_f_in,
                "n_samples_out": dataset.n_samples, "n_features_out": dataset.n_features,
            })

        # Fit drift detector on the post-reduction feature matrix
        if self.drift_detector is not None:
            self.drift_detector.fit(dataset)

        # Cost / qubit warning after all reductions are applied
        if self.encoder is not None:
            import warnings

            from quprep.validation.cost import estimate_cost
            from quprep.validation.input_validator import QuPrepWarning
            cost = estimate_cost(self.encoder, dataset.n_features)
            if cost.warning:
                warnings.warn(cost.warning, QuPrepWarning, stacklevel=4)
            self._last_cost = cost
        else:
            self._last_cost = None

        self._last_audit_log = audit if audit else None
        return dataset

    def _apply_stages(self, dataset) -> PipelineResult:
        """Apply fitted stages to dataset and return PipelineResult."""
        audit: list[dict] = []

        if self.preprocessor is not None:
            _preprocessors = (
                self.preprocessor
                if isinstance(self.preprocessor, list)
                else [self.preprocessor]
            )
            for i, pre in enumerate(_preprocessors):
                n_s_in, n_f_in = dataset.n_samples, dataset.n_features
                dataset = pre.transform(dataset)
                label = "preprocessor" if len(_preprocessors) == 1 else f"preprocessor[{i}]"
                audit.append({
                    "stage": label,
                    "n_samples_in": n_s_in, "n_features_in": n_f_in,
                    "n_samples_out": dataset.n_samples, "n_features_out": dataset.n_features,
                })

        if self.cleaner is not None:
            n_s_in, n_f_in = dataset.n_samples, dataset.n_features
            dataset = self.cleaner.transform(dataset)
            audit.append({
                "stage": "cleaner",
                "n_samples_in": n_s_in, "n_features_in": n_f_in,
                "n_samples_out": dataset.n_samples, "n_features_out": dataset.n_features,
            })

        if self.reducer is not None:
            n_s_in, n_f_in = dataset.n_samples, dataset.n_features
            dataset = self.reducer.transform(dataset)
            audit.append({
                "stage": "reducer",
                "n_samples_in": n_s_in, "n_features_in": n_f_in,
                "n_samples_out": dataset.n_samples, "n_features_out": dataset.n_features,
            })

        # Check drift against training distribution (post-reduction, pre-normalization —
        # same point in the pipeline where the detector was fitted)
        if self.drift_detector is not None:
            self._last_drift_report = self.drift_detector.check(dataset)
        else:
            self._last_drift_report = None

        if self._resolved_normalizer is not None:
            n_s_in, n_f_in = dataset.n_samples, dataset.n_features
            dataset = self._resolved_normalizer.transform(dataset)
            audit.append({
                "stage": "normalizer",
                "n_samples_in": n_s_in, "n_features_in": n_f_in,
                "n_samples_out": dataset.n_samples, "n_features_out": dataset.n_features,
            })

        self._last_audit_log = audit if audit else None
        return self._encode_export(dataset)

    def _encode_export(self, dataset) -> PipelineResult:
        """Run encoder + exporter on an already-transformed dataset."""
        drift = self._last_drift_report

        if self.encoder is None:
            return PipelineResult(
                dataset=dataset, encoded=None, circuits=None,
                cost=None, audit_log=self._last_audit_log, drift_report=drift,
            )

        encoded = self.encoder.encode_batch(dataset)

        if self.exporter is None:
            return PipelineResult(
                dataset=dataset, encoded=encoded, circuits=None,
                cost=self._last_cost, audit_log=self._last_audit_log, drift_report=drift,
            )

        circuits = self.exporter.export_batch(encoded)
        return PipelineResult(
            dataset=dataset, encoded=encoded, circuits=circuits,
            cost=self._last_cost, audit_log=self._last_audit_log, drift_report=drift,
        )

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
            import scipy.sparse as _sp
            if _sp.issparse(source):
                from quprep.ingest.numpy_ingester import NumpyIngester
                return NumpyIngester().load(source)
        except ImportError:
            pass

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
