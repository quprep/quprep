"""Tests for the quprep CLI."""

from __future__ import annotations

import pytest

from quprep.cli import build_parser, main

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def csv_file(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("x0,x1,x2,x3\n0.1,0.2,0.3,0.4\n0.5,0.6,0.7,0.8\n0.9,0.1,0.2,0.3\n")
    return str(f)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class TestParser:
    def test_version_flag(self, capsys):
        with pytest.raises(SystemExit) as exc:
            build_parser().parse_args(["--version"])
        assert exc.value.code == 0

    def test_no_command_returns_zero(self):
        assert main([]) == 0

    def test_convert_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["convert", "data.csv"])
        assert args.encoding == "angle"
        assert args.framework == "qasm"
        assert args.rotation == "ry"
        assert args.output is None
        assert args.samples is None

    def test_convert_all_flags(self):
        parser = build_parser()
        args = parser.parse_args([
            "convert", "data.csv",
            "--encoding", "basis",
            "--framework", "qasm",
            "--output", "out.qasm",
            "--samples", "2",
        ])
        assert args.encoding == "basis"
        assert args.output == "out.qasm"
        assert args.samples == 2

    def test_recommend_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["recommend", "data.csv"])
        assert args.task == "classification"
        assert args.qubits is None

    def test_invalid_encoding_rejected(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["convert", "data.csv", "--encoding", "banana"])

    def test_invalid_framework_rejected(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["convert", "data.csv", "--framework", "braket"])


# ---------------------------------------------------------------------------
# convert — phase 2 stubs
# ---------------------------------------------------------------------------

class TestConvertPhase2Stubs:
    def test_iqp_encoding_works(self, csv_file, capsys):
        rc = main(["convert", csv_file, "--encoding", "iqp"])
        assert rc == 0

    def test_pennylane_framework_rejected(self, csv_file, capsys):
        rc = main(["convert", csv_file, "--framework", "pennylane"])
        assert rc == 1
        assert "v0.2.0" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# convert — angle encoding, qasm framework
# ---------------------------------------------------------------------------

class TestConvertAngleQASM:
    def test_returns_zero(self, csv_file):
        assert main(["convert", csv_file]) == 0

    def test_stdout_contains_qasm(self, csv_file, capsys):
        main(["convert", csv_file])
        out = capsys.readouterr().out
        assert "OPENQASM 3.0" in out

    def test_stdout_has_ry_gates(self, csv_file, capsys):
        main(["convert", csv_file])
        out = capsys.readouterr().out
        assert "ry(" in out

    def test_rx_rotation_flag(self, csv_file, capsys):
        main(["convert", csv_file, "--rotation", "rx"])
        out = capsys.readouterr().out
        assert "rx(" in out

    def test_output_file_written(self, csv_file, tmp_path):
        out_file = str(tmp_path / "circuit.qasm")
        rc = main(["convert", csv_file, "--output", out_file])
        assert rc == 0
        from pathlib import Path
        content = Path(out_file).read_text()
        assert "OPENQASM 3.0" in content

    def test_output_file_message(self, csv_file, tmp_path, capsys):
        out_file = str(tmp_path / "circuit.qasm")
        main(["convert", csv_file, "--output", out_file])
        out = capsys.readouterr().out
        assert "circuit" in out.lower()

    def test_samples_flag_limits_output(self, csv_file, capsys):
        main(["convert", csv_file, "--samples", "1"])
        out = capsys.readouterr().out
        # Only 1 sample → only 1 OPENQASM header
        assert out.count("OPENQASM 3.0") == 1

    def test_all_samples_by_default(self, csv_file, capsys):
        main(["convert", csv_file])
        out = capsys.readouterr().out
        # CSV has 3 rows → 3 circuits
        assert out.count("OPENQASM 3.0") == 3


# ---------------------------------------------------------------------------
# convert — basis encoding
# ---------------------------------------------------------------------------

class TestConvertBasisQASM:
    def test_basis_returns_zero(self, csv_file):
        assert main(["convert", csv_file, "--encoding", "basis"]) == 0

    def test_basis_stdout_is_qasm(self, csv_file, capsys):
        main(["convert", csv_file, "--encoding", "basis"])
        out = capsys.readouterr().out
        assert "OPENQASM 3.0" in out


# ---------------------------------------------------------------------------
# convert — error handling
# ---------------------------------------------------------------------------

class TestConvertErrors:
    def test_missing_file_returns_one(self, capsys):
        rc = main(["convert", "/nonexistent/file.csv"])
        assert rc == 1
        assert "not found" in capsys.readouterr().err.lower()

    def test_missing_qiskit_returns_one(self, csv_file, capsys):
        try:
            import qiskit  # noqa: F401
            pytest.skip("qiskit installed")
        except ImportError:
            rc = main(["convert", csv_file, "--framework", "qiskit"])
            assert rc == 1
            assert "dependency" in capsys.readouterr().err.lower()


# ---------------------------------------------------------------------------
# recommend
# ---------------------------------------------------------------------------

class TestRecommend:
    def test_recommend_returns_one(self, csv_file, capsys):
        rc = main(["recommend", csv_file])
        assert rc == 1
        assert "v0.2.0" in capsys.readouterr().err
