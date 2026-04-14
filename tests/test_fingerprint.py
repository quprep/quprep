"""Tests for reproducibility fingerprinting."""

import json

import pytest

import quprep as qd
from quprep.core.fingerprint import (
    FingerprintResult,
    _make_serializable,
    _stage_config,
    fingerprint_pipeline,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline(**kwargs):
    return qd.Pipeline(**kwargs)


# ---------------------------------------------------------------------------
# FingerprintResult
# ---------------------------------------------------------------------------


class TestFingerprintResult:
    def test_repr(self):
        fp = fingerprint_pipeline(_make_pipeline(encoder=qd.AngleEncoder()))
        r = repr(fp)
        assert r.startswith("FingerprintResult(hash=sha256:")
        assert "stages=" in r

    def test_to_dict_has_hash_and_timestamp(self):
        fp = fingerprint_pipeline(_make_pipeline(encoder=qd.AngleEncoder()))
        d = fp.to_dict()
        assert d["hash"].startswith("sha256:")
        assert "timestamp" in d

    def test_to_json_is_valid_json(self):
        fp = fingerprint_pipeline(_make_pipeline(encoder=qd.AngleEncoder()))
        parsed = json.loads(fp.to_json())
        assert "hash" in parsed
        assert "stages" in parsed

    def test_str_returns_json(self):
        fp = fingerprint_pipeline(_make_pipeline(encoder=qd.AngleEncoder()))
        assert json.loads(str(fp))  # must be valid JSON

    def test_save_json(self, tmp_path):
        fp = fingerprint_pipeline(_make_pipeline(encoder=qd.AngleEncoder()))
        out = tmp_path / "fp.json"
        fp.save(str(out), format="json")
        assert out.exists()
        parsed = json.loads(out.read_text())
        assert "hash" in parsed

    def test_save_invalid_format(self, tmp_path):
        fp = fingerprint_pipeline(_make_pipeline(encoder=qd.AngleEncoder()))
        with pytest.raises(ValueError, match="format must be"):
            fp.save(str(tmp_path / "fp.txt"), format="xml")

    def test_save_creates_parent_dirs(self, tmp_path):
        fp = fingerprint_pipeline(_make_pipeline(encoder=qd.AngleEncoder()))
        out = tmp_path / "nested" / "dir" / "fp.json"
        fp.save(str(out))
        assert out.exists()

    def test_to_yaml_returns_string(self, tmp_path):
        pytest.importorskip("yaml", reason="pyyaml not installed")
        fp = fingerprint_pipeline(_make_pipeline(encoder=qd.AngleEncoder()))
        yml = fp.to_yaml()
        assert isinstance(yml, str)
        assert "sha256:" in yml

    def test_save_yaml(self, tmp_path):
        pytest.importorskip("yaml", reason="pyyaml not installed")
        fp = fingerprint_pipeline(_make_pipeline(encoder=qd.AngleEncoder()))
        out = tmp_path / "fp.yaml"
        fp.save(str(out), format="yaml")
        assert out.exists()
        assert "sha256:" in out.read_text()

    def test_to_yaml_missing_pyyaml(self):
        import sys
        fp = fingerprint_pipeline(_make_pipeline(encoder=qd.AngleEncoder()))
        sentinel = object()
        old = sys.modules.get("yaml", sentinel)
        sys.modules["yaml"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="pyyaml"):
                fp.to_yaml()
        finally:
            if old is sentinel:
                sys.modules.pop("yaml", None)
            else:
                sys.modules["yaml"] = old


# ---------------------------------------------------------------------------
# Hash determinism
# ---------------------------------------------------------------------------


class TestHashDeterminism:
    def test_same_config_same_hash(self):
        p1 = _make_pipeline(encoder=qd.AngleEncoder(rotation="ry"))
        p2 = _make_pipeline(encoder=qd.AngleEncoder(rotation="ry"))
        assert fingerprint_pipeline(p1).hash == fingerprint_pipeline(p2).hash

    def test_different_rotation_different_hash(self):
        p1 = _make_pipeline(encoder=qd.AngleEncoder(rotation="ry"))
        p2 = _make_pipeline(encoder=qd.AngleEncoder(rotation="rx"))
        assert fingerprint_pipeline(p1).hash != fingerprint_pipeline(p2).hash

    def test_different_encoder_different_hash(self):
        p1 = _make_pipeline(encoder=qd.AngleEncoder())
        p2 = _make_pipeline(encoder=qd.AmplitudeEncoder())
        assert fingerprint_pipeline(p1).hash != fingerprint_pipeline(p2).hash

    def test_empty_pipeline_hash_is_stable(self):
        p = _make_pipeline()
        h1 = fingerprint_pipeline(p).hash
        h2 = fingerprint_pipeline(p).hash
        assert h1 == h2

    def test_adding_reducer_changes_hash(self):
        p1 = _make_pipeline(encoder=qd.AngleEncoder())
        p2 = _make_pipeline(encoder=qd.AngleEncoder(), reducer=qd.PCAReducer(n_components=4))
        assert fingerprint_pipeline(p1).hash != fingerprint_pipeline(p2).hash

    def test_hash_not_in_config_dict(self):
        # config itself must not contain 'hash' — only to_dict() adds it
        fp = fingerprint_pipeline(_make_pipeline(encoder=qd.AngleEncoder()))
        assert "hash" not in fp.config


# ---------------------------------------------------------------------------
# Config structure
# ---------------------------------------------------------------------------


class TestConfigStructure:
    def test_config_has_required_top_level_keys(self):
        fp = fingerprint_pipeline(_make_pipeline(encoder=qd.AngleEncoder()))
        assert "quprep_version" in fp.config
        assert "python_version" in fp.config
        assert "stages" in fp.config
        assert "dependencies" in fp.config

    def test_encoder_stage_captured(self):
        fp = fingerprint_pipeline(_make_pipeline(encoder=qd.AngleEncoder()))
        assert "encoder" in fp.config["stages"]
        assert fp.config["stages"]["encoder"]["class"] == "AngleEncoder"

    def test_empty_pipeline_has_no_stages(self):
        fp = fingerprint_pipeline(_make_pipeline())
        assert fp.config["stages"] == {}

    def test_multi_stage_pipeline(self):
        p = _make_pipeline(
            cleaner=qd.Imputer(),
            reducer=qd.PCAReducer(n_components=4),
            encoder=qd.AmplitudeEncoder(),
            exporter=qd.QASMExporter(),
        )
        stages = fingerprint_pipeline(p).config["stages"]
        assert "cleaner" in stages
        assert "reducer" in stages
        assert "encoder" in stages
        assert "exporter" in stages

    def test_preprocessor_list_captured(self):
        p = _make_pipeline(preprocessor=[qd.WindowTransformer()])
        stages = fingerprint_pipeline(p).config["stages"]
        assert "preprocessor" in stages
        assert isinstance(stages["preprocessor"], list)

    def test_encoder_params_captured(self):
        p = _make_pipeline(encoder=qd.AngleEncoder(rotation="rz"))
        params = fingerprint_pipeline(p).config["stages"]["encoder"]["params"]
        assert params.get("rotation") == "rz"

    def test_dependencies_is_dict(self):
        fp = fingerprint_pipeline(_make_pipeline())
        assert isinstance(fp.config["dependencies"], dict)
        # numpy should always be present
        assert "numpy" in fp.config["dependencies"]

    def test_python_version_is_string(self):
        fp = fingerprint_pipeline(_make_pipeline())
        assert isinstance(fp.config["python_version"], str)


# ---------------------------------------------------------------------------
# Pipeline.fingerprint() method
# ---------------------------------------------------------------------------


class TestPipelineMethod:
    def test_method_returns_fingerprint_result(self):
        p = _make_pipeline(encoder=qd.AngleEncoder())
        fp = p.fingerprint()
        assert isinstance(fp, FingerprintResult)

    def test_method_matches_standalone_function(self):
        p = _make_pipeline(encoder=qd.AngleEncoder())
        assert p.fingerprint().hash == fingerprint_pipeline(p).hash


# ---------------------------------------------------------------------------
# Top-level namespace
# ---------------------------------------------------------------------------


class TestTopLevelExport:
    def test_fingerprint_pipeline_accessible(self):
        assert callable(qd.fingerprint_pipeline)

    def test_fingerprint_result_accessible(self):
        assert qd.FingerprintResult is FingerprintResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_make_serializable_primitives(self):
        assert _make_serializable(42) == 42
        assert _make_serializable(3.14) == 3.14
        assert _make_serializable("hello") == "hello"
        assert _make_serializable(None) is None
        assert _make_serializable(True) is True

    def test_make_serializable_list(self):
        assert _make_serializable([1, 2, 3]) == [1, 2, 3]

    def test_make_serializable_dict(self):
        assert _make_serializable({"a": 1}) == {"a": 1}

    def test_make_serializable_tuple(self):
        result = _make_serializable((1, 2))
        assert result == [1, 2]

    def test_make_serializable_numpy_array(self):
        import numpy as np

        result = _make_serializable(np.array([1.0, 2.0]))
        assert result == [1.0, 2.0]

    def test_make_serializable_numpy_scalar(self):
        import numpy as np

        result = _make_serializable(np.float32(3.14))
        assert isinstance(result, float)

    def test_make_serializable_unknown_object(self):
        class Foo:
            pass

        result = _make_serializable(Foo())
        assert result == "<Foo>"

    def test_make_serializable_nested_get_params(self):
        # Object with get_params → should be serialized as class+params dict
        class FakeEstimator:
            def get_params(self):
                return {"alpha": 0.5, "beta": "x"}

        obj = FakeEstimator()
        result = _make_serializable(obj)
        assert isinstance(result, dict)
        assert result["class"] == "FakeEstimator"
        assert "params" in result
        assert result["params"]["alpha"] == 0.5

    def test_stage_config_no_get_params(self):
        # Object without get_params → falls back to inspect.signature
        class SimpleStage:
            def __init__(self, alpha=1.0, beta="x"):
                self.alpha = alpha
                self.beta = beta

        s = SimpleStage(alpha=2.0)
        config = _stage_config(s)
        assert config["class"] == "SimpleStage"
        assert config["params"]["alpha"] == 2.0
        assert config["params"]["beta"] == "x"

    def test_fingerprint_missing_optional_packages(self):
        # Some tracked packages (like cirq-core) won't be installed →
        # PackageNotFoundError branch is hit, package absent from dependencies
        fp = fingerprint_pipeline(_make_pipeline())
        deps = fp.config["dependencies"]
        # numpy is always present; an uninstalled optional pkg should be absent
        assert "numpy" in deps
        # cirq-core is optional — if absent, it should not appear in deps
        import importlib.metadata
        try:
            importlib.metadata.version("cirq-core")
        except importlib.metadata.PackageNotFoundError:
            assert "cirq-core" not in deps

    def test_stage_config_has_class_module_params(self):
        stage = qd.AngleEncoder(rotation="rz")
        config = _stage_config(stage)
        assert config["class"] == "AngleEncoder"
        assert "module" in config
        assert "params" in config

    def test_stage_config_params_serializable(self):
        stage = qd.PCAReducer(n_components=3)
        config = _stage_config(stage)
        # All params must be JSON-serializable
        json.dumps(config)
