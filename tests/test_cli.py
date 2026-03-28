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

    def test_pennylane_framework_missing_dep(self, csv_file, capsys):
        try:
            import pennylane  # noqa: F401
            import pytest
            pytest.skip("pennylane installed")
        except ImportError:
            rc = main(["convert", csv_file, "--framework", "pennylane"])
            assert rc == 1
            assert "Missing dependency" in capsys.readouterr().err


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
    def test_recommend_returns_zero(self, csv_file, capsys):
        rc = main(["recommend", csv_file])
        assert rc == 0

    def test_recommend_prints_encoding(self, csv_file, capsys):
        main(["recommend", csv_file])
        out = capsys.readouterr().out
        assert "Recommended encoding" in out

    def test_recommend_with_task(self, csv_file, capsys):
        rc = main(["recommend", csv_file, "--task", "kernel"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "iqp" in out

    def test_recommend_with_qubits(self, csv_file, capsys):
        rc = main(["recommend", csv_file, "--qubits", "4"])
        assert rc == 0

    def test_recommend_missing_file(self, capsys):
        rc = main(["recommend", "nonexistent.csv"])
        assert rc == 1
        assert "File not found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# quprep validate
# ---------------------------------------------------------------------------

class TestValidateCommand:
    def test_validate_basic(self, csv_file, capsys):
        rc = main(["validate", csv_file])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Shape" in out
        assert "3 samples" in out

    def test_validate_prints_columns(self, csv_file, capsys):
        main(["validate", csv_file])
        out = capsys.readouterr().out
        assert "x0" in out

    def test_validate_no_nan(self, csv_file, capsys):
        main(["validate", csv_file])
        out = capsys.readouterr().out
        assert "none" in out  # "NaN     : none"

    def test_validate_reports_nan(self, tmp_path, capsys):
        f = tmp_path / "nan.csv"
        f.write_text("a,b\n1.0,2.0\n,3.0\n1.0,\n")
        rc = main(["validate", str(f)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "missing values" in out

    def test_validate_missing_file(self, capsys):
        rc = main(["validate", "nonexistent.csv"])
        assert rc == 1
        assert "File not found" in capsys.readouterr().err

    def test_validate_with_valid_schema(self, tmp_path, capsys):
        import json
        csv_f = tmp_path / "data.csv"
        csv_f.write_text("age,score\n25.0,0.8\n30.0,0.9\n")
        schema_f = tmp_path / "schema.json"
        schema_f.write_text(json.dumps([
            {"name": "age", "dtype": "continuous", "min_value": 0, "max_value": 120},
            {"name": "score", "dtype": "continuous", "min_value": 0.0, "max_value": 1.0},
        ]))
        rc = main(["validate", str(csv_f), "--schema", str(schema_f)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "OK" in out

    def test_validate_with_violated_schema(self, tmp_path, capsys):
        import json
        csv_f = tmp_path / "data.csv"
        csv_f.write_text("age,score\n-5.0,0.8\n30.0,0.9\n")
        schema_f = tmp_path / "schema.json"
        schema_f.write_text(json.dumps([
            {"name": "age", "dtype": "continuous", "min_value": 0},
            {"name": "score", "dtype": "continuous"},
        ]))
        rc = main(["validate", str(csv_f), "--schema", str(schema_f)])
        assert rc == 1
        assert "FAILED" in capsys.readouterr().err

    def test_validate_parser_defaults(self):
        args = build_parser().parse_args(["validate", "data.csv"])
        assert args.source == "data.csv"
        assert args.schema is None
        assert args.infer_schema is None

    def test_validate_infer_schema_to_stdout(self, csv_file, capsys):
        rc = main(["validate", csv_file, "--infer-schema", "-"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "x0" in out  # column name appears in inferred schema JSON

    def test_validate_infer_schema_to_file(self, csv_file, tmp_path, capsys):
        import json
        out_path = str(tmp_path / "schema.json")
        rc = main(["validate", csv_file, "--infer-schema", out_path])
        assert rc == 0
        data = json.loads(open(out_path).read())
        assert isinstance(data, list)
        assert data[0]["name"] == "x0"

    def test_validate_infer_then_validate(self, csv_file, tmp_path):
        schema_path = str(tmp_path / "schema.json")
        main(["validate", csv_file, "--infer-schema", schema_path])
        # inferred schema should validate the same file cleanly
        rc = main(["validate", csv_file, "--schema", schema_path])
        assert rc == 0
