"""Tests for the plugin registry (register_encoder / register_exporter)."""

from __future__ import annotations

import numpy as np
import pytest

from quprep.encode.base import BaseEncoder, EncodedResult
from quprep.plugins import (
    get_encoder_class,
    get_exporter_class,
    list_encoders,
    list_exporters,
    register_encoder,
    register_exporter,
    unregister_encoder,
    unregister_exporter,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def cleanup_registry():
    """Remove any test entries from the registry after each test."""
    yield
    for name in ["_test_enc", "_test_exp", "_test_enc2", "_test_exp2"]:
        unregister_encoder(name)
        unregister_exporter(name)


def _make_encoder_cls(name="_test_enc"):
    @register_encoder(name)
    class DummyEncoder(BaseEncoder):
        @property
        def n_qubits(self):
            return None

        @property
        def depth(self):
            return 1

        def encode(self, x: np.ndarray) -> EncodedResult:
            return EncodedResult(
                parameters=x.copy(),
                metadata={"encoding": name, "n_qubits": len(x), "depth": 1},
            )

    return DummyEncoder


def _make_exporter_cls(name="_test_exp"):
    @register_exporter(name)
    class DummyExporter:
        def export(self, encoded):
            return f"DUMMY:{len(encoded.parameters)}"

        def export_batch(self, encoded_list):
            return [self.export(e) for e in encoded_list]

    return DummyExporter


# ---------------------------------------------------------------------------
# register_encoder
# ---------------------------------------------------------------------------

class TestRegisterEncoder:
    def test_register_and_retrieve(self):
        cls = _make_encoder_cls("_test_enc")
        assert get_encoder_class("_test_enc") is cls

    def test_registered_name_in_list(self):
        _make_encoder_cls("_test_enc")
        assert "_test_enc" in list_encoders()

    def test_duplicate_raises(self):
        _make_encoder_cls("_test_enc")
        with pytest.raises(ValueError, match="already registered"):
            _make_encoder_cls("_test_enc")

    def test_unknown_returns_none(self):
        assert get_encoder_class("no_such_encoder_xyz") is None

    def test_unregister_removes(self):
        _make_encoder_cls("_test_enc")
        unregister_encoder("_test_enc")
        assert get_encoder_class("_test_enc") is None

    def test_unregister_missing_is_noop(self):
        unregister_encoder("never_registered_abc")  # should not raise

    def test_registered_encoder_is_functional(self):
        cls = _make_encoder_cls("_test_enc")
        enc = cls()
        x = np.array([1.0, 2.0, 3.0])
        result = enc.encode(x)
        np.testing.assert_array_equal(result.parameters, x)
        assert result.metadata["encoding"] == "_test_enc"


# ---------------------------------------------------------------------------
# register_exporter
# ---------------------------------------------------------------------------

class TestRegisterExporter:
    def test_register_and_retrieve(self):
        cls = _make_exporter_cls("_test_exp")
        assert get_exporter_class("_test_exp") is cls

    def test_registered_name_in_list(self):
        _make_exporter_cls("_test_exp")
        assert "_test_exp" in list_exporters()

    def test_duplicate_raises(self):
        _make_exporter_cls("_test_exp")
        with pytest.raises(ValueError, match="already registered"):
            _make_exporter_cls("_test_exp")

    def test_unknown_returns_none(self):
        assert get_exporter_class("no_such_exporter_xyz") is None

    def test_unregister_removes(self):
        _make_exporter_cls("_test_exp")
        unregister_exporter("_test_exp")
        assert get_exporter_class("_test_exp") is None

    def test_registered_exporter_is_functional(self):
        cls = _make_exporter_cls("_test_exp")
        exp = cls()
        from quprep.encode.base import EncodedResult
        encoded = EncodedResult(
            parameters=np.ones(5),
            metadata={"encoding": "_test_exp", "n_qubits": 5},
        )
        assert exp.export(encoded) == "DUMMY:5"


# ---------------------------------------------------------------------------
# Integration with prepare()
# ---------------------------------------------------------------------------

class TestPluginIntegrationWithPrepare:
    def test_plugin_encoder_works_with_prepare(self):
        # Plugin encoder that wraps angle encoding so QASMExporter can handle it
        import quprep
        from quprep.encode.angle import AngleEncoder

        @register_encoder("_test_enc")
        class AngleWrapEncoder(BaseEncoder):
            @property
            def n_qubits(self):
                return None

            @property
            def depth(self):
                return 1

            def encode(self, x):
                return AngleEncoder().encode(x)

        X = np.random.default_rng(0).random((5, 3))
        result = quprep.prepare(X, encoding="_test_enc", framework="qasm")
        assert result is not None
        unregister_encoder("_test_enc")

    def test_unknown_encoding_raises_in_prepare(self):
        import quprep
        X = np.ones((3, 2))
        with pytest.raises(ValueError, match="Unknown encoding"):
            quprep.prepare(X, encoding="completely_unknown_xyz")

    def test_unknown_framework_raises_in_prepare(self):
        import quprep
        X = np.ones((3, 2))
        with pytest.raises(ValueError, match="Unknown framework"):
            quprep.prepare(X, framework="completely_unknown_xyz")


# ---------------------------------------------------------------------------
# list_encoders / list_exporters
# ---------------------------------------------------------------------------

class TestListFunctions:
    def test_list_encoders_returns_sorted_list(self):
        result = list_encoders()
        assert isinstance(result, list)
        assert result == sorted(result)

    def test_list_exporters_returns_sorted_list(self):
        result = list_exporters()
        assert isinstance(result, list)
        assert result == sorted(result)

    def test_list_encoders_grows_on_register(self):
        before = len(list_encoders())
        _make_encoder_cls("_test_enc")
        after = len(list_encoders())
        assert after == before + 1

    def test_list_exporters_grows_on_register(self):
        before = len(list_exporters())
        _make_exporter_cls("_test_exp")
        after = len(list_exporters())
        assert after == before + 1
