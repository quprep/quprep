"""Tests for suggest_qubits and QubitSuggestion."""

from __future__ import annotations

import numpy as np
import pytest

from quprep.core.qubit_suggestion import QubitSuggestion, suggest_qubits


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _arr(n_samples=20, n_features=6):
    rng = np.random.default_rng(0)
    return rng.uniform(0.0, 1.0, size=(n_samples, n_features))


# ---------------------------------------------------------------------------
# Return type and basic shape
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_qubit_suggestion(self):
        result = suggest_qubits(_arr())
        assert isinstance(result, QubitSuggestion)

    def test_n_features_matches_dataset(self):
        arr = _arr(n_features=5)
        result = suggest_qubits(arr)
        assert result.n_features == 5

    def test_n_qubits_leq_n_features(self):
        arr = _arr(n_features=8)
        result = suggest_qubits(arr)
        assert result.n_qubits <= arr.shape[1]

    def test_n_qubits_leq_ceiling(self):
        arr = _arr(n_features=30)
        result = suggest_qubits(arr, max_qubits=10)
        assert result.n_qubits <= 10


# ---------------------------------------------------------------------------
# Qubit count logic
# ---------------------------------------------------------------------------

class TestQubitCount:
    def test_features_within_budget_uses_all_features(self):
        arr = _arr(n_features=6)
        result = suggest_qubits(arr, max_qubits=20)
        assert result.n_qubits == 6

    def test_features_exceed_budget_caps_at_ceiling(self):
        arr = _arr(n_features=50)
        result = suggest_qubits(arr, max_qubits=10)
        assert result.n_qubits == 10

    def test_features_exactly_at_ceiling(self):
        arr = _arr(n_features=20)
        result = suggest_qubits(arr)
        assert result.n_qubits == 20

    def test_default_ceiling_is_20(self):
        arr = _arr(n_features=25)
        result = suggest_qubits(arr)
        assert result.n_qubits == 20

    def test_custom_max_qubits_respected(self):
        arr = _arr(n_features=4)
        result = suggest_qubits(arr, max_qubits=2)
        assert result.n_qubits == 2


# ---------------------------------------------------------------------------
# NISQ-safe flag
# ---------------------------------------------------------------------------

class TestNisqSafe:
    def test_nisq_safe_when_few_features(self):
        arr = _arr(n_features=8)
        result = suggest_qubits(arr)
        assert result.nisq_safe is True

    def test_nisq_safe_when_at_ceiling(self):
        arr = _arr(n_features=20)
        result = suggest_qubits(arr)
        assert result.nisq_safe is True

    def test_not_nisq_safe_when_max_qubits_above_20(self):
        arr = _arr(n_features=30)
        result = suggest_qubits(arr, max_qubits=25)
        assert result.nisq_safe is False


# ---------------------------------------------------------------------------
# Warning
# ---------------------------------------------------------------------------

class TestWarning:
    def test_no_warning_when_features_fit(self):
        arr = _arr(n_features=5)
        result = suggest_qubits(arr)
        assert result.warning is None

    def test_warning_when_reduction_needed(self):
        arr = _arr(n_features=30)
        result = suggest_qubits(arr, max_qubits=10)
        assert result.warning is not None
        assert "PCAReducer" in result.warning


# ---------------------------------------------------------------------------
# Encoding hints per task
# ---------------------------------------------------------------------------

class TestEncodingHint:
    def test_qaoa_suggests_basis(self):
        arr = _arr(n_features=5)
        result = suggest_qubits(arr, task="qaoa")
        assert result.encoding_hint == "basis"

    def test_kernel_small_suggests_iqp(self):
        arr = _arr(n_features=6)
        result = suggest_qubits(arr, task="kernel")
        assert result.encoding_hint == "iqp"

    def test_kernel_large_suggests_angle(self):
        arr = _arr(n_features=15)
        result = suggest_qubits(arr, task="kernel")
        assert result.encoding_hint == "angle"

    def test_simulation_suggests_hamiltonian(self):
        arr = _arr(n_features=5)
        result = suggest_qubits(arr, task="simulation")
        assert result.encoding_hint == "hamiltonian"

    def test_large_sample_avoids_amplitude(self):
        arr = _arr(n_samples=600, n_features=4)
        result = suggest_qubits(arr, task="classification")
        assert result.encoding_hint == "angle"

    def test_small_sample_small_qubits_may_suggest_amplitude(self):
        arr = _arr(n_samples=20, n_features=3)
        result = suggest_qubits(arr, task="classification")
        # Either amplitude or angle — just ensure it's a valid string
        assert result.encoding_hint in {"amplitude", "angle"}

    def test_default_classification_is_angle(self):
        arr = _arr(n_samples=200, n_features=8)
        result = suggest_qubits(arr)
        assert result.encoding_hint == "angle"


# ---------------------------------------------------------------------------
# Invalid task
# ---------------------------------------------------------------------------

class TestInvalidTask:
    def test_invalid_task_raises(self):
        arr = _arr()
        with pytest.raises(ValueError, match="Unknown task"):
            suggest_qubits(arr, task="foobar")


# ---------------------------------------------------------------------------
# Source types
# ---------------------------------------------------------------------------

class TestSourceTypes:
    def test_accepts_ndarray(self):
        result = suggest_qubits(_arr())
        assert result.n_features == 6

    def test_accepts_list(self):
        data = [[0.1, 0.2, 0.3]] * 10
        result = suggest_qubits(data)
        assert result.n_features == 3

    def test_accepts_dataframe(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(_arr())
        result = suggest_qubits(df)
        assert result.n_features == 6

    def test_accepts_dataset(self):
        from quprep.core.dataset import Dataset
        ds = Dataset(
            data=_arr(),
            feature_names=["f1", "f2", "f3", "f4", "f5", "f6"],
            feature_types=["continuous"] * 6,
        )
        result = suggest_qubits(ds)
        assert result.n_features == 6

    def test_invalid_source_raises(self):
        with pytest.raises(TypeError):
            suggest_qubits(42)


# ---------------------------------------------------------------------------
# String representations
# ---------------------------------------------------------------------------

class TestStringRepr:
    def test_str_contains_suggested_qubits(self):
        arr = _arr(n_features=6)
        result = suggest_qubits(arr)
        text = str(result)
        assert "Suggested qubits" in text
        assert str(result.n_qubits) in text

    def test_str_contains_warning_when_present(self):
        arr = _arr(n_features=30)
        result = suggest_qubits(arr, max_qubits=5)
        assert "Warning" in str(result)

    def test_str_no_warning_line_when_absent(self):
        arr = _arr(n_features=5)
        result = suggest_qubits(arr)
        assert "Warning" not in str(result)

    def test_repr_compact(self):
        result = suggest_qubits(_arr())
        r = repr(result)
        assert "QubitSuggestion" in r
        assert "n_qubits" in r

    def test_reasoning_is_non_empty(self):
        result = suggest_qubits(_arr())
        assert result.reasoning.strip() != ""


# ---------------------------------------------------------------------------
# Top-level import
# ---------------------------------------------------------------------------

def test_top_level_import():
    import quprep as qd
    assert hasattr(qd, "suggest_qubits")
    assert hasattr(qd, "QubitSuggestion")
    result = qd.suggest_qubits(_arr())
    assert isinstance(result, qd.QubitSuggestion)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def test_cli_suggest_basic(self, tmp_path):
        import csv

        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["a", "b", "c"])
            for _ in range(10):
                writer.writerow([0.1, 0.2, 0.3])

        from quprep.cli import main
        rc = main(["suggest", str(csv_file)])
        assert rc == 0

    def test_cli_suggest_task_and_max_qubits(self, tmp_path):
        import csv

        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["a", "b", "c", "d", "e"])
            for _ in range(15):
                writer.writerow([0.1, 0.2, 0.3, 0.4, 0.5])

        from quprep.cli import main
        rc = main(["suggest", str(csv_file), "--task", "kernel", "--max-qubits", "4"])
        assert rc == 0

    def test_cli_suggest_missing_file(self):
        from quprep.cli import main
        rc = main(["suggest", "/nonexistent/data.csv"])
        assert rc == 1

    def test_cli_suggest_invalid_task(self, tmp_path):
        # argparse will reject unknown task choices before cmd_suggest runs
        import csv

        csv_file = tmp_path / "data.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["a"])
            writer.writerow([1.0])
        from quprep.cli import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["suggest", str(csv_file), "--task", "invalid"])
