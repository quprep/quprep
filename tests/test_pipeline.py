"""Tests for the Pipeline orchestrator and prepare() one-liner."""

from __future__ import annotations

import numpy as np
import pytest

from quprep.core.dataset import Dataset
from quprep.core.pipeline import Pipeline, PipelineResult
from quprep.encode.amplitude import AmplitudeEncoder
from quprep.encode.angle import AngleEncoder
from quprep.encode.base import EncodedResult
from quprep.encode.basis import BasisEncoder
from quprep.export.qasm_export import QASMExporter
from quprep.validation import CostEstimate

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_array():
    rng = np.random.default_rng(42)
    return rng.uniform(0.0, 1.0, size=(5, 4))


@pytest.fixture
def simple_dataset(simple_array):
    return Dataset(
        data=simple_array,
        feature_names=["a", "b", "c", "d"],
        feature_types=["continuous"] * 4,
    )


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------

class TestPipelineResult:
    def test_circuit_property_returns_first_circuit(self):
        circuits = ["circuit_0", "circuit_1", "circuit_2"]
        result = PipelineResult(dataset=None, encoded=None, circuits=circuits)
        assert result.circuit == "circuit_0"

    def test_circuit_property_falls_back_to_encoded(self):
        enc = [EncodedResult(np.array([1.0]), metadata={})]
        result = PipelineResult(dataset=None, encoded=enc, circuits=None)
        assert result.circuit is enc[0]

    def test_circuit_property_none_when_both_absent(self):
        result = PipelineResult(dataset=None, encoded=None, circuits=None)
        assert result.circuit is None

    def test_repr(self):
        enc = [EncodedResult(np.array([1.0]), metadata={}) for _ in range(3)]
        result = PipelineResult(dataset=None, encoded=enc, circuits=["c"] * 3)
        assert "3" in repr(result)
        assert "yes" in repr(result)

    def test_repr_no_circuits(self):
        enc = [EncodedResult(np.array([1.0]), metadata={})]
        result = PipelineResult(dataset=None, encoded=enc, circuits=None)
        assert "no" in repr(result)

    def test_cost_defaults_none(self):
        result = PipelineResult(dataset=None, encoded=None, circuits=None)
        assert result.cost is None

    def test_audit_log_defaults_none(self):
        result = PipelineResult(dataset=None, encoded=None, circuits=None)
        assert result.audit_log is None

    def test_repr_includes_nisq_safe_when_cost_present(self):
        from quprep.validation import CostEstimate
        cost = CostEstimate(
            encoding="angle", n_features=4, n_qubits=4,
            gate_count=4, circuit_depth=1, two_qubit_gates=0,
            nisq_safe=True, warning=None,
        )
        enc = [EncodedResult(np.array([1.0]), metadata={})]
        result = PipelineResult(dataset=None, encoded=enc, circuits=None, cost=cost)
        assert "nisq_safe=True" in repr(result)


# ---------------------------------------------------------------------------
# Pipeline — ingestion
# ---------------------------------------------------------------------------

class TestPipelineIngestion:
    def test_dataset_passthrough(self, simple_dataset):
        p = Pipeline(encoder=AngleEncoder())
        result = p.fit_transform(simple_dataset)
        assert isinstance(result, PipelineResult)

    def test_numpy_array_auto_ingested(self, simple_array):
        p = Pipeline(encoder=AngleEncoder())
        result = p.fit_transform(simple_array)
        assert isinstance(result, PipelineResult)
        assert result.encoded is not None

    def test_list_of_lists_auto_ingested(self):
        data = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        p = Pipeline(encoder=AngleEncoder())
        result = p.fit_transform(data)
        assert len(result.encoded) == 2

    def test_csv_file_auto_ingested(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("x0,x1,x2\n0.1,0.2,0.3\n0.4,0.5,0.6\n")
        p = Pipeline(encoder=AngleEncoder())
        result = p.fit_transform(str(csv_file))
        assert len(result.encoded) == 2

    def test_path_object_auto_ingested(self, tmp_path):
        from pathlib import Path
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("x0,x1\n0.1,0.2\n0.3,0.4\n")
        p = Pipeline(encoder=AngleEncoder())
        result = p.fit_transform(Path(csv_file))
        assert len(result.encoded) == 2

    def test_unsupported_type_raises(self):
        p = Pipeline(encoder=AngleEncoder())
        with pytest.raises(TypeError, match="Cannot ingest"):
            p.fit_transform(12345)

    def test_explicit_ingester_used(self, simple_array):
        from quprep.ingest.numpy_ingester import NumpyIngester
        p = Pipeline(ingester=NumpyIngester(), encoder=AngleEncoder())
        result = p.fit_transform(simple_array)
        assert result.encoded is not None


# ---------------------------------------------------------------------------
# Pipeline — no encoder / no exporter
# ---------------------------------------------------------------------------

class TestPipelinePartial:
    def test_no_encoder_returns_dataset(self, simple_dataset):
        p = Pipeline()
        result = p.fit_transform(simple_dataset)
        assert result.dataset is not None
        assert result.encoded is None
        assert result.circuits is None

    def test_no_exporter_returns_encoded_list(self, simple_dataset):
        p = Pipeline(encoder=AngleEncoder())
        result = p.fit_transform(simple_dataset)
        assert result.encoded is not None
        assert result.circuits is None
        assert len(result.encoded) == simple_dataset.n_samples

    def test_fit_returns_self(self, simple_dataset):
        p = Pipeline()
        assert p.fit(simple_dataset) is p

    def test_transform_requires_fit(self, simple_dataset):
        p = Pipeline()
        with pytest.raises(RuntimeError, match="not been fitted"):
            p.transform(simple_dataset)


# ---------------------------------------------------------------------------
# Pipeline — full end-to-end
# ---------------------------------------------------------------------------

class TestPipelineEndToEnd:
    def test_angle_qasm_pipeline(self, simple_array):
        p = Pipeline(encoder=AngleEncoder(), exporter=QASMExporter())
        result = p.fit_transform(simple_array)
        assert result.circuits is not None
        assert len(result.circuits) == 5
        assert all(isinstance(c, str) for c in result.circuits)
        assert all("OPENQASM 3.0" in c for c in result.circuits)

    def test_basis_qasm_pipeline(self, simple_array):
        p = Pipeline(encoder=BasisEncoder(), exporter=QASMExporter())
        result = p.fit_transform(simple_array)
        assert len(result.circuits) == 5

    def test_circuit_property_is_first(self, simple_array):
        p = Pipeline(encoder=AngleEncoder(), exporter=QASMExporter())
        result = p.fit_transform(simple_array)
        assert result.circuit == result.circuits[0]

    def test_n_encoded_equals_n_samples(self, simple_dataset):
        p = Pipeline(encoder=AngleEncoder())
        result = p.fit_transform(simple_dataset)
        assert len(result.encoded) == simple_dataset.n_samples

    def test_encoded_n_qubits_equals_n_features(self, simple_dataset):
        p = Pipeline(encoder=AngleEncoder())
        result = p.fit_transform(simple_dataset)
        for enc in result.encoded:
            assert enc.metadata["n_qubits"] == simple_dataset.n_features


# ---------------------------------------------------------------------------
# Pipeline — auto-normalization
# ---------------------------------------------------------------------------

class TestPipelineAutoNormalization:
    def test_angle_ry_auto_normalizes_to_pi(self, simple_array):
        """Angle Ry: data should land in [0, π] after pipeline normalization."""
        p = Pipeline(encoder=AngleEncoder(rotation="ry"))
        result = p.fit_transform(simple_array)
        for enc in result.encoded:
            assert np.all(enc.parameters >= 0.0 - 1e-9)
            assert np.all(enc.parameters <= np.pi + 1e-9)

    def test_angle_rx_auto_normalizes_to_pm_pi(self, simple_array):
        """Angle Rx: data should land in [−π, π] after pipeline normalization."""
        p = Pipeline(encoder=AngleEncoder(rotation="rx"))
        result = p.fit_transform(simple_array)
        for enc in result.encoded:
            assert np.all(enc.parameters >= -np.pi - 1e-9)
            assert np.all(enc.parameters <= np.pi + 1e-9)

    def test_amplitude_auto_normalizes_l2(self, simple_array):
        """Amplitude: each sample should have unit L2 norm."""
        p = Pipeline(encoder=AmplitudeEncoder())
        result = p.fit_transform(simple_array)
        for enc in result.encoded:
            assert np.isclose(np.linalg.norm(enc.parameters), 1.0, atol=1e-10)

    def test_basis_auto_normalizes_binary(self, simple_array):
        """Basis: each sample should be binary {0, 1}."""
        p = Pipeline(encoder=BasisEncoder())
        result = p.fit_transform(simple_array)
        for enc in result.encoded:
            assert set(np.unique(enc.parameters)).issubset({0.0, 1.0})

    def test_explicit_normalizer_overrides_auto(self, simple_array):
        """When normalizer is explicitly set, auto-selection is skipped."""
        from quprep.normalize.scalers import Scaler
        p = Pipeline(
            encoder=AngleEncoder(rotation="ry"),
            normalizer=Scaler("zscore"),
        )
        result = p.fit_transform(simple_array)
        # zscore data may go outside [0,π] — just confirm the override ran
        assert result.encoded is not None


# ---------------------------------------------------------------------------
# Pipeline — with cleaner
# ---------------------------------------------------------------------------

class TestPipelineWithCleaner:
    def test_cleaner_applied(self, tmp_path):
        """Pipeline applies imputer before encoding."""

        from quprep.clean.imputer import Imputer
        from quprep.ingest.numpy_ingester import NumpyIngester

        data = np.array([
            [1.0, 2.0, 3.0],
            [np.nan, 2.0, 3.0],
            [1.0, np.nan, 3.0],
        ])
        p = Pipeline(
            ingester=NumpyIngester(),
            cleaner=Imputer(strategy="mean"),
            encoder=AngleEncoder(),
        )
        result = p.fit_transform(data)
        # No NaN should remain in parameters
        for enc in result.encoded:
            assert not np.any(np.isnan(enc.parameters))


# ---------------------------------------------------------------------------
# prepare() one-liner
# ---------------------------------------------------------------------------

class TestPrepare:
    def test_prepare_returns_pipeline_result(self, simple_array):
        import quprep
        result = quprep.prepare(simple_array, encoding="angle", framework="qasm")
        assert isinstance(result, PipelineResult)

    def test_prepare_angle_qasm(self, simple_array):
        import quprep
        result = quprep.prepare(simple_array)
        assert result.circuits is not None
        assert all("OPENQASM 3.0" in c for c in result.circuits)

    def test_prepare_basis_qasm(self, simple_array):
        import quprep
        result = quprep.prepare(simple_array, encoding="basis", framework="qasm")
        assert result.circuits is not None

    def test_prepare_unknown_encoding_raises(self, simple_array):
        import quprep
        with pytest.raises(ValueError, match="Unknown encoding"):
            quprep.prepare(simple_array, encoding="totally_unknown")

    def test_prepare_unknown_framework_raises(self, simple_array):
        import quprep
        with pytest.raises(ValueError, match="Unknown framework"):
            quprep.prepare(simple_array, framework="braket")

    def test_prepare_rotation_kwarg(self, simple_array):
        import quprep
        result = quprep.prepare(simple_array, encoding="angle", rotation="rx")
        assert all("rx(" in c for c in result.circuits)

    def test_prepare_circuit_property(self, simple_array):
        import quprep
        result = quprep.prepare(simple_array)
        assert result.circuit is not None
        assert isinstance(result.circuit, str)

    def test_prepare_qiskit_missing_raises(self, simple_array):
        """prepare() with framework='qiskit' should raise ImportError if qiskit absent."""
        try:
            import qiskit  # noqa: F401
            pytest.skip("qiskit installed")
        except ImportError:
            import quprep
            with pytest.raises(ImportError, match="pip install quprep"):
                quprep.prepare(simple_array, framework="qiskit")

    def test_version_accessible(self):
        import quprep
        assert quprep.__version__ == "0.4.0"


# ---------------------------------------------------------------------------
# PipelineResult — cost attribute
# ---------------------------------------------------------------------------

class TestPipelineResultCost:
    def test_cost_is_cost_estimate(self, simple_array):
        p = Pipeline(encoder=AngleEncoder())
        result = p.fit_transform(simple_array)
        assert isinstance(result.cost, CostEstimate)

    def test_cost_encoding_name(self, simple_array):
        result = Pipeline(encoder=AngleEncoder()).fit_transform(simple_array)
        assert result.cost.encoding == "angle"

    def test_cost_n_qubits(self, simple_dataset):
        result = Pipeline(encoder=AngleEncoder()).fit_transform(simple_dataset)
        assert result.cost.n_qubits == simple_dataset.n_features

    def test_cost_none_when_no_encoder(self, simple_dataset):
        result = Pipeline().fit_transform(simple_dataset)
        assert result.cost is None

    def test_cost_preserved_through_transform(self, simple_array):
        p = Pipeline(encoder=AngleEncoder())
        p.fit(simple_array)
        result = p.transform(simple_array)
        # cost is computed at fit time and re-used for transform
        assert isinstance(result.cost, CostEstimate)

    def test_cost_nisq_safe_small_dataset(self, simple_array):
        result = Pipeline(encoder=AngleEncoder()).fit_transform(simple_array)
        assert result.cost.nisq_safe is True


# ---------------------------------------------------------------------------
# PipelineResult — audit_log attribute
# ---------------------------------------------------------------------------

class TestPipelineResultAuditLog:
    def test_audit_log_none_when_no_stages_run(self, simple_dataset):
        # No encoder → no auto-normalizer → no stages → audit_log is None
        result = Pipeline().fit_transform(simple_dataset)
        assert result.audit_log is None

    def test_audit_log_has_auto_normalizer_entry(self, simple_dataset):
        # AngleEncoder triggers auto-normalization → normalizer audit entry present
        result = Pipeline(encoder=AngleEncoder()).fit_transform(simple_dataset)
        assert result.audit_log is not None
        stages = [e["stage"] for e in result.audit_log]
        assert "normalizer" in stages

    def test_audit_log_has_cleaner_entry(self, simple_dataset):
        from quprep.clean.imputer import Imputer
        result = Pipeline(cleaner=Imputer(), encoder=AngleEncoder()).fit_transform(simple_dataset)
        assert result.audit_log is not None
        stages = [e["stage"] for e in result.audit_log]
        assert "cleaner" in stages

    def test_audit_log_has_normalizer_entry_when_explicit(self, simple_dataset):
        from quprep.normalize.scalers import Scaler
        result = Pipeline(
            normalizer=Scaler("minmax"), encoder=AngleEncoder()
        ).fit_transform(simple_dataset)
        assert result.audit_log is not None
        stages = [e["stage"] for e in result.audit_log]
        assert "normalizer" in stages

    def test_audit_log_entry_keys(self, simple_dataset):
        from quprep.clean.imputer import Imputer
        result = Pipeline(cleaner=Imputer(), encoder=AngleEncoder()).fit_transform(simple_dataset)
        entry = result.audit_log[0]
        for key in ("stage", "n_samples_in", "n_features_in", "n_samples_out", "n_features_out"):
            assert key in entry

    def test_audit_log_cleaner_drops_rows(self):
        """OutlierHandler (clip) should not drop rows — shape preserved."""
        from quprep.clean.outlier import OutlierHandler
        rng = np.random.default_rng(0)
        ds = Dataset(
            data=rng.random((20, 3)).astype(np.float64),
            feature_names=["a", "b", "c"],
            feature_types=["continuous"] * 3,
        )
        result = Pipeline(cleaner=OutlierHandler(), encoder=AngleEncoder()).fit_transform(ds)
        entry = next(e for e in result.audit_log if e["stage"] == "cleaner")
        assert entry["n_samples_in"] == 20
        assert entry["n_samples_out"] == 20

    def test_audit_log_reducer_reduces_features(self, simple_dataset):
        from quprep.reduce.pca import PCAReducer
        result = Pipeline(
            reducer=PCAReducer(n_components=2), encoder=AngleEncoder()
        ).fit_transform(simple_dataset)
        entry = next(e for e in result.audit_log if e["stage"] == "reducer")
        assert entry["n_features_in"] == simple_dataset.n_features
        assert entry["n_features_out"] == 2

    def test_audit_log_available_after_transform(self, simple_dataset):
        from quprep.clean.imputer import Imputer
        p = Pipeline(cleaner=Imputer(), encoder=AngleEncoder())
        p.fit(simple_dataset)
        result = p.transform(simple_dataset)
        assert result.audit_log is not None
        assert result.audit_log[0]["stage"] == "cleaner"


# ---------------------------------------------------------------------------
# Pipeline.summary()
# ---------------------------------------------------------------------------

class TestPipelineSummary:
    def test_summary_returns_string(self, simple_dataset):
        p = Pipeline(encoder=AngleEncoder())
        p.fit(simple_dataset)
        assert isinstance(p.summary(), str)

    def test_summary_shows_fitted(self, simple_dataset):
        p = Pipeline(encoder=AngleEncoder())
        p.fit(simple_dataset)
        assert "yes" in p.summary()

    def test_summary_shows_not_fitted(self):
        p = Pipeline(encoder=AngleEncoder())
        assert "no" in p.summary()

    def test_summary_shows_encoder_name(self, simple_dataset):
        p = Pipeline(encoder=AngleEncoder())
        p.fit(simple_dataset)
        assert "AngleEncoder" in p.summary()

    def test_summary_shows_cost(self, simple_dataset):
        p = Pipeline(encoder=AngleEncoder())
        p.fit(simple_dataset)
        assert "angle" in p.summary()
        assert "qubits" in p.summary()

    def test_summary_str_dunder(self, simple_dataset):
        p = Pipeline(encoder=AngleEncoder())
        p.fit(simple_dataset)
        assert str(p) == p.summary()

    def test_summary_omits_cost_when_not_fitted(self):
        p = Pipeline(encoder=AngleEncoder())
        assert "cost" not in p.summary()

    def test_summary_shows_schema_when_set(self, simple_dataset):
        from quprep.validation import DataSchema, FeatureSpec
        schema = DataSchema([FeatureSpec(f"f{i}", "continuous") for i in range(4)])
        p = Pipeline(encoder=AngleEncoder(), schema=schema)
        assert "schema" in p.summary()
        assert "4" in p.summary()


# ---------------------------------------------------------------------------
# PipelineResult.summary()
# ---------------------------------------------------------------------------

class TestPipelineResultSummary:
    def test_summary_returns_string(self, simple_dataset):
        result = Pipeline(encoder=AngleEncoder()).fit_transform(simple_dataset)
        assert isinstance(result.summary(), str)

    def test_summary_shows_samples(self, simple_dataset):
        result = Pipeline(encoder=AngleEncoder()).fit_transform(simple_dataset)
        assert str(simple_dataset.n_samples) in result.summary()

    def test_summary_shows_cost_section(self, simple_dataset):
        result = Pipeline(encoder=AngleEncoder()).fit_transform(simple_dataset)
        assert "Cost estimate" in result.summary()
        assert "gate count" in result.summary()

    def test_summary_shows_audit_table(self, simple_dataset):
        from quprep.clean.imputer import Imputer
        result = Pipeline(cleaner=Imputer(), encoder=AngleEncoder()).fit_transform(simple_dataset)
        assert "Preprocessing stages" in result.summary()
        assert "cleaner" in result.summary()

    def test_summary_no_cost_section_without_encoder(self, simple_dataset):
        result = Pipeline().fit_transform(simple_dataset)
        assert "Cost estimate" not in result.summary()

    def test_summary_no_audit_table_without_stages(self, simple_dataset):
        result = Pipeline().fit_transform(simple_dataset)
        assert "Preprocessing stages" not in result.summary()


# ---------------------------------------------------------------------------
# import quprep as qd alias
# ---------------------------------------------------------------------------

class TestQdAlias:
    def test_qd_pipeline(self):
        import quprep as qd
        assert qd.Pipeline is Pipeline

    def test_qd_angle_encoder(self):
        import quprep as qd
        from quprep.encode.angle import AngleEncoder as _AE
        assert qd.AngleEncoder is _AE

    def test_qd_imputer(self):
        import quprep as qd
        from quprep.clean.imputer import Imputer as _I
        assert qd.Imputer is _I

    def test_qd_pca_reducer(self):
        import quprep as qd
        from quprep.reduce.pca import PCAReducer as _PCA
        assert qd.PCAReducer is _PCA

    def test_qd_scaler(self):
        import quprep as qd
        from quprep.normalize.scalers import Scaler as _S
        assert qd.Scaler is _S

    def test_qd_qasm_exporter(self):
        import quprep as qd
        from quprep.export.qasm_export import QASMExporter as _QE
        assert qd.QASMExporter is _QE

    def test_qd_full_pipeline_usage(self, simple_array):
        import quprep as qd
        result = qd.Pipeline(encoder=qd.AngleEncoder()).fit_transform(simple_array)
        assert isinstance(result, qd.PipelineResult)

    def test_qd_schema_classes(self):
        import quprep as qd
        assert qd.DataSchema is not None
        assert qd.FeatureSpec is not None
        assert qd.SchemaViolationError is not None
