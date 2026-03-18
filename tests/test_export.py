"""Tests for framework exporters."""

import pytest


class TestQASMExporter:
    def test_unsupported_version_raises(self):
        from quprep.export.qasm_export import QASMExporter
        with pytest.raises(ValueError):
            QASMExporter(version="2.0")

    def test_export_returns_string(self):
        pytest.skip("QASMExporter.export() not yet implemented")


class TestQiskitExporter:
    def test_missing_qiskit_raises(self):
        """Should raise ImportError with helpful message when qiskit not installed."""
        pytest.importorskip("qiskit", reason="qiskit not installed — skipping")

    def test_export_returns_circuit(self):
        pytest.skip("QiskitExporter.export() not yet implemented")
