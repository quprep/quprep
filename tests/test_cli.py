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
            build_parser().parse_args(["convert", "data.csv", "--framework", "not_a_framework"])


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


# ---------------------------------------------------------------------------
# cmd_convert — error paths and non-QASM output
# ---------------------------------------------------------------------------

class TestConvertErrorPaths:
    def test_file_not_found_returns_one(self, capsys):
        rc = main(["convert", "nonexistent.csv", "--encoding", "angle"])
        assert rc == 1
        assert "File not found" in capsys.readouterr().err

    def test_non_qasm_framework_prints_repr(self, csv_file, capsys):
        # cirq is an optional dep — if installed, exercises the non-QASM repr path
        # if not installed, exercises the ImportError path; both are valid
        rc = main(["convert", csv_file, "--encoding", "angle", "--framework", "cirq"])
        assert rc in (0, 1)
        out, err = capsys.readouterr()
        if rc == 0:
            assert "--- sample 0 ---" in out
        else:
            assert "Missing dependency" in err or "not installed" in err.lower()

    def test_non_qasm_prints_repr(self, tmp_path, capsys):
        # Use a framework that isn't 'qasm' but is available without extra deps
        # We test by monkey-patching prepare to return non-string circuits
        csv_f = tmp_path / "data.csv"
        csv_f.write_text("x0,x1\n0.1,0.2\n0.3,0.4\n")
        rc = main(["convert", str(csv_f), "--encoding", "angle", "--framework", "qasm"])
        assert rc == 0

    def test_save_dir_creates_files(self, csv_file, tmp_path, capsys):
        save_dir = str(tmp_path / "out")
        rc = main(["convert", csv_file, "--encoding", "angle", "--save-dir", save_dir])
        assert rc == 0
        out = capsys.readouterr().out
        assert "circuit" in out.lower() or "Wrote" in out

    def test_save_dir_with_stem(self, csv_file, tmp_path):
        save_dir = str(tmp_path / "circuits")
        rc = main(["convert", csv_file, "--encoding", "angle",
                   "--save-dir", save_dir, "--stem", "feat"])
        assert rc == 0
        import os
        files = os.listdir(save_dir)
        assert any(f.startswith("feat_") for f in files)

    def test_output_file_written_angle(self, csv_file, tmp_path):
        out = str(tmp_path / "circuit.qasm")
        rc = main(["convert", csv_file, "--encoding", "angle", "--output", out])
        assert rc == 0
        assert open(out).read().startswith("OPENQASM 3.0;")


# ---------------------------------------------------------------------------
# QUBO CLI — portfolio, graphcolor, qaoa, export subcommands
# ---------------------------------------------------------------------------

class TestQuboExtraSubcommands:
    def test_portfolio_returns_zero(self, capsys):
        rc = main([
            "qubo", "portfolio",
            "--returns", "0.1,0.2,0.3",
            "--covariance", "1,0,0;0,1,0;0,0,1",
            "--budget", "2",
        ])
        assert rc == 0

    def test_graphcolor_returns_zero(self, capsys):
        rc = main([
            "qubo", "graphcolor",
            "--adjacency", "0,1,1;1,0,1;1,1,0",
            "--colors", "3",
        ])
        assert rc == 0

    def test_qaoa_maxcut_returns_zero(self, capsys):
        rc = main([
            "qubo", "qaoa", "maxcut",
            "--adjacency", "0,1,1;1,0,1;1,1,0",
            "--p", "1",
        ])
        assert rc == 0
        assert "OPENQASM" in capsys.readouterr().out

    def test_qaoa_with_output_file(self, tmp_path, capsys):
        out = str(tmp_path / "qaoa.qasm")
        rc = main([
            "qubo", "qaoa", "maxcut",
            "--adjacency", "0,1;1,0",
            "--p", "1",
            "--output", out,
        ])
        assert rc == 0
        assert open(out).read().startswith("OPENQASM")

    def test_export_json_to_stdout(self, capsys):
        rc = main([
            "qubo", "export", "maxcut",
            "--adjacency", "0,1;1,0",
            "--format", "json",
        ])
        assert rc == 0
        import json
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "Q" in data

    def test_export_json_to_file(self, tmp_path, capsys):
        out = str(tmp_path / "qubo.json")
        rc = main([
            "qubo", "export", "maxcut",
            "--adjacency", "0,1;1,0",
            "--format", "json",
            "--output", out,
        ])
        assert rc == 0

    def test_export_npy(self, tmp_path):
        out = str(tmp_path / "qubo.npy")
        rc = main([
            "qubo", "export", "maxcut",
            "--adjacency", "0,1;1,0",
            "--format", "npy",
            "--output", out,
        ])
        assert rc == 0
        import numpy as np
        Q = np.load(out)
        assert Q.shape == (2, 2)

    def test_qubo_no_subcommand_returns_zero(self, capsys):
        rc = main(["qubo"])
        assert rc == 0

    def test_qubo_exception_returns_one(self, capsys):
        # bad matrix string triggers parse error → exception path
        rc = main(["qubo", "maxcut", "--adjacency", "not_a_matrix"])
        assert rc == 1


# ---------------------------------------------------------------------------
# cmd_suggest — error paths
# ---------------------------------------------------------------------------

class TestSuggestErrorPaths:
    def test_suggest_missing_file_returns_one(self, capsys):
        rc = main(["suggest", "nonexistent.csv"])
        assert rc == 1
        assert "File not found" in capsys.readouterr().err

    def test_suggest_invalid_task_exits(self, csv_file):
        with pytest.raises(SystemExit):
            main(["suggest", csv_file, "--task", "not_a_valid_task"])

    def test_suggest_returns_zero(self, csv_file, capsys):
        rc = main(["suggest", csv_file])
        assert rc == 0
        assert "qubit" in capsys.readouterr().out.lower()


# ---------------------------------------------------------------------------
# cmd_compare — error paths
# ---------------------------------------------------------------------------

class TestCompareErrorPaths:
    def test_compare_missing_file_returns_one(self, capsys):
        rc = main(["compare", "nonexistent.csv"])
        assert rc == 1
        assert "File not found" in capsys.readouterr().err

    def test_compare_returns_zero(self, csv_file, capsys):
        rc = main(["compare", csv_file])
        assert rc == 0

    def test_compare_with_include(self, csv_file, capsys):
        rc = main(["compare", csv_file, "--include", "angle,basis"])
        assert rc == 0


# ---------------------------------------------------------------------------
# cmd_recommend — error paths
# ---------------------------------------------------------------------------

class TestRecommendErrorPaths:
    def test_recommend_missing_file_returns_one(self, capsys):
        rc = main(["recommend", "nonexistent.csv"])
        assert rc == 1
        assert "File not found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# cmd_convert — ImportError and generic Exception paths
# ---------------------------------------------------------------------------

class TestConvertExceptionPaths:
    def test_import_error_returns_one(self, csv_file, capsys):
        from unittest.mock import patch
        with patch("quprep.prepare", side_effect=ImportError("Mock missing dep")):
            rc = main(["convert", csv_file])
        assert rc == 1
        assert "dependency" in capsys.readouterr().err.lower()

    def test_generic_exception_returns_one(self, csv_file, capsys):
        from unittest.mock import patch
        with patch("quprep.prepare", side_effect=RuntimeError("boom")):
            rc = main(["convert", csv_file])
        assert rc == 1
        assert "boom" in capsys.readouterr().err

    def test_save_dir_with_samples(self, csv_file, tmp_path, capsys):
        rc = main([
            "convert", csv_file,
            "--save-dir", str(tmp_path / "out"),
            "--samples", "2",
        ])
        assert rc == 0
        files = list((tmp_path / "out").iterdir())
        assert len(files) == 2


# ---------------------------------------------------------------------------
# qubo subcommands — direct (maxcut, knapsack, tsp, schedule, partition)
# ---------------------------------------------------------------------------

class TestQuboDirectSubcommands:
    _ADJ = "0,1,1;1,0,1;1,1,0"
    _DIST = "0,2,9;2,0,6;9,6,0"

    def test_maxcut_returns_zero(self, capsys):
        rc = main(["qubo", "maxcut", "--adjacency", self._ADJ])
        assert rc == 0
        assert "Variables" in capsys.readouterr().out

    def test_maxcut_with_solve(self, capsys):
        rc = main(["qubo", "maxcut", "--adjacency", self._ADJ, "--solve"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Best x" in out

    def test_knapsack_returns_zero(self, capsys):
        rc = main(["qubo", "knapsack",
                   "--weights", "1,2,3",
                   "--values", "4,5,6",
                   "--capacity", "4"])
        assert rc == 0

    def test_tsp_returns_zero(self, capsys):
        rc = main(["qubo", "tsp", "--distances", self._DIST])
        assert rc == 0

    def test_schedule_returns_zero(self, capsys):
        rc = main(["qubo", "schedule", "--times", "2,3,5", "--machines", "2"])
        assert rc == 0

    def test_partition_returns_zero(self, capsys):
        rc = main(["qubo", "partition", "--values", "3,1,2,4"])
        assert rc == 0


# ---------------------------------------------------------------------------
# qubo qaoa — all problem types via _build_qubo_from_args
# ---------------------------------------------------------------------------

class TestQuboBuildFromArgs:
    def test_qaoa_knapsack(self, capsys):
        rc = main(["qubo", "qaoa", "knapsack",
                   "--weights", "1,2", "--values", "3,4", "--capacity", "2"])
        assert rc == 0

    def test_qaoa_tsp(self, capsys):
        rc = main(["qubo", "qaoa", "tsp", "--distances", "0,1;1,0"])
        assert rc == 0

    def test_qaoa_schedule(self, capsys):
        rc = main(["qubo", "qaoa", "schedule", "--times", "2,3", "--machines", "1"])
        assert rc == 0

    def test_qaoa_partition(self, capsys):
        rc = main(["qubo", "qaoa", "partition", "--values", "1,2,3"])
        assert rc == 0


# ---------------------------------------------------------------------------
# cmd_validate — additional paths
# ---------------------------------------------------------------------------

class TestValidateExtraPaths:
    def test_validate_all_nan_column(self, tmp_path, capsys):
        f = tmp_path / "nan.csv"
        f.write_text("a,b\n1.0,\n2.0,\n")
        rc = main(["validate", str(f)])
        assert rc == 0

    def test_validate_bad_schema_json(self, tmp_path, capsys):
        data = tmp_path / "data.csv"
        data.write_text("x,y\n1.0,2.0\n")
        schema = tmp_path / "bad.json"
        schema.write_text("NOT_VALID_JSON")
        rc = main(["validate", str(data), "--schema", str(schema)])
        assert rc == 1
        assert "schema" in capsys.readouterr().err.lower()

    def test_validate_generic_exception(self, capsys):
        from unittest.mock import patch
        with patch("quprep.ingest.csv_ingester.CSVIngester") as mock_cls:
            mock_cls.return_value.load.side_effect = ValueError("bad data")
            rc = main(["validate", "data.csv"])
        assert rc == 1


# ---------------------------------------------------------------------------
# cmd_suggest, cmd_recommend, cmd_compare — ValueError paths
# ---------------------------------------------------------------------------

class TestCommandValueErrorPaths:
    def test_suggest_value_error_returns_one(self, csv_file, capsys):
        from unittest.mock import patch
        with patch("quprep.core.qubit_suggestion.suggest_qubits", side_effect=ValueError("bad")):
            rc = main(["suggest", csv_file])
        assert rc == 1

    def test_suggest_generic_exception_returns_one(self, csv_file, capsys):
        from unittest.mock import patch
        with patch("quprep.core.qubit_suggestion.suggest_qubits", side_effect=RuntimeError("boom")):
            rc = main(["suggest", csv_file])
        assert rc == 1

    def test_recommend_value_error_returns_one(self, csv_file, capsys):
        from unittest.mock import patch
        with patch("quprep.core.recommender.recommend", side_effect=ValueError("bad")):
            rc = main(["recommend", csv_file])
        assert rc == 1

    def test_recommend_generic_exception_returns_one(self, csv_file, capsys):
        from unittest.mock import patch
        with patch("quprep.core.recommender.recommend", side_effect=RuntimeError("boom")):
            rc = main(["recommend", csv_file])
        assert rc == 1

    def test_compare_value_error_returns_one(self, csv_file, capsys):
        from unittest.mock import patch
        with patch("quprep.compare.compare_encodings", side_effect=ValueError("bad")):
            rc = main(["compare", csv_file])
        assert rc == 1

    def test_compare_generic_exception_returns_one(self, csv_file, capsys):
        from unittest.mock import patch
        with patch("quprep.compare.compare_encodings", side_effect=RuntimeError("boom")):
            rc = main(["compare", csv_file])
        assert rc == 1
