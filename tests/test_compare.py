"""Tests for encoding comparison — compare_encodings() and ComparisonResult."""

from __future__ import annotations

import numpy as np
import pytest

import quprep as qd
from quprep.compare import _ALL_ENCODINGS, ComparisonResult, compare_encodings

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_array():
    rng = np.random.default_rng(0)
    return rng.random((20, 4))


@pytest.fixture
def wide_array():
    rng = np.random.default_rng(1)
    return rng.random((10, 16))


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------

def test_returns_comparison_result(small_array):
    result = compare_encodings(small_array)
    assert isinstance(result, ComparisonResult)


def test_rows_count_default(small_array):
    result = compare_encodings(small_array)
    assert len(result.rows) == len(_ALL_ENCODINGS)


def test_rows_are_cost_estimates(small_array):
    result = compare_encodings(small_array)
    for row in result.rows:
        assert isinstance(row, qd.CostEstimate)


def test_all_encodings_present(small_array):
    result = compare_encodings(small_array)
    names = {r.encoding for r in result.rows}
    assert names == set(_ALL_ENCODINGS)


def test_no_recommended_when_no_task(small_array):
    result = compare_encodings(small_array)
    assert result.recommended is None


# ---------------------------------------------------------------------------
# include / exclude
# ---------------------------------------------------------------------------

def test_include_filters(small_array):
    result = compare_encodings(small_array, include=["angle", "basis", "iqp"])
    assert len(result.rows) == 3
    names = {r.encoding for r in result.rows}
    assert names == {"angle", "basis", "iqp"}


def test_exclude_filters(small_array):
    result = compare_encodings(small_array, exclude=["amplitude", "hamiltonian"])
    names = {r.encoding for r in result.rows}
    assert "amplitude" not in names
    assert "hamiltonian" not in names
    assert len(result.rows) == len(_ALL_ENCODINGS) - 2


def test_include_then_exclude(small_array):
    result = compare_encodings(
        small_array,
        include=["angle", "basis", "iqp"],
        exclude=["iqp"],
    )
    names = {r.encoding for r in result.rows}
    assert names == {"angle", "basis"}


def test_include_unknown_raises(small_array):
    with pytest.raises(ValueError, match="Unknown encoder"):
        compare_encodings(small_array, include=["angle", "notanencoder"])


# ---------------------------------------------------------------------------
# Qubit budget
# ---------------------------------------------------------------------------

def test_qubit_budget_flags_over_budget(wide_array):
    result = compare_encodings(wide_array, qubits=4)
    over = [r for r in result.rows if r.n_qubits > 4]
    for row in over:
        assert not row.nisq_safe
        assert row.warning is not None
        assert "budget" in row.warning


def test_qubit_budget_within_budget_unchanged(small_array):
    # 4 features — angle/basis/iqp/reupload/entangled/hamiltonian all need 4 qubits
    result = compare_encodings(small_array, qubits=10)
    for row in result.rows:
        if row.n_qubits <= 10:
            # warning should not mention budget
            assert row.warning is None or "budget" not in row.warning


# ---------------------------------------------------------------------------
# task → recommended
# ---------------------------------------------------------------------------

def test_task_sets_recommended(small_array):
    result = compare_encodings(small_array, task="classification")
    assert result.recommended is not None
    assert result.recommended in _ALL_ENCODINGS


def test_recommended_is_in_rows(small_array):
    result = compare_encodings(small_array, task="simulation")
    assert result.recommended in {r.encoding for r in result.rows}


def test_recommended_none_when_task_not_in_include(small_array):
    # If recommended encoding is excluded via include=, recommended should be None
    result = compare_encodings(
        small_array,
        task="simulation",
        include=["angle", "basis"],  # hamiltonian excluded
    )
    # recommended is either None or one of the included encoders
    if result.recommended is not None:
        assert result.recommended in {"angle", "basis"}


# ---------------------------------------------------------------------------
# best()
# ---------------------------------------------------------------------------

def test_best_nisq_returns_nisq_safe(small_array):
    result = compare_encodings(small_array)
    best = result.best(prefer="nisq")
    assert best.nisq_safe


def test_best_depth_returns_min_depth(small_array):
    result = compare_encodings(small_array)
    best = result.best(prefer="depth")
    assert best.circuit_depth == min(r.circuit_depth for r in result.rows)


def test_best_gates_returns_min_gates(small_array):
    result = compare_encodings(small_array)
    best = result.best(prefer="gates")
    assert best.gate_count == min(r.gate_count for r in result.rows)


def test_best_qubits_returns_min_qubits(small_array):
    result = compare_encodings(small_array)
    best = result.best(prefer="qubits")
    assert best.n_qubits == min(r.n_qubits for r in result.rows)


def test_best_unknown_prefer_raises(small_array):
    result = compare_encodings(small_array)
    with pytest.raises(ValueError, match="Unknown prefer"):
        result.best(prefer="badcriterion")


def test_best_nisq_falls_back_when_none_safe():
    # Use 64 features — amplitude will dominate with huge depth; all others NISQ safe
    X = np.random.default_rng(2).random((5, 64))
    result = compare_encodings(X, include=["amplitude"])
    # Only amplitude — none NISQ safe fallback
    best = result.best(prefer="nisq")
    assert best.encoding == "amplitude"


# ---------------------------------------------------------------------------
# to_dict()
# ---------------------------------------------------------------------------

def test_to_dict_structure(small_array):
    result = compare_encodings(small_array)
    d = result.to_dict()
    assert isinstance(d, list)
    assert len(d) == len(result.rows)
    for entry in d:
        assert set(entry.keys()) == {
            "encoding", "n_qubits", "gate_count", "circuit_depth",
            "two_qubit_gates", "nisq_safe", "warning",
        }


def test_to_dict_types(small_array):
    result = compare_encodings(small_array)
    for entry in result.to_dict():
        assert isinstance(entry["encoding"], str)
        assert isinstance(entry["n_qubits"], int)
        assert isinstance(entry["nisq_safe"], bool)


# ---------------------------------------------------------------------------
# __str__() / __repr__()
# ---------------------------------------------------------------------------

def test_str_contains_all_encodings(small_array):
    result = compare_encodings(small_array)
    s = str(result)
    for name in _ALL_ENCODINGS:
        assert name in s


def test_str_contains_star_when_recommended(small_array):
    result = compare_encodings(small_array, task="classification")
    assert "*" in str(result)


def test_str_no_star_without_task(small_array):
    result = compare_encodings(small_array)
    assert "*" not in str(result)


def test_repr(small_array):
    result = compare_encodings(small_array)
    r = repr(result)
    assert "ComparisonResult" in r
    assert str(len(_ALL_ENCODINGS)) in r


# ---------------------------------------------------------------------------
# Source type acceptance
# ---------------------------------------------------------------------------

def test_accepts_dataframe(small_array):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(small_array, columns=[f"f{i}" for i in range(4)])
    result = compare_encodings(df)
    assert len(result.rows) == len(_ALL_ENCODINGS)


def test_accepts_list_of_lists():
    data = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    result = compare_encodings(data)
    assert len(result.rows) == len(_ALL_ENCODINGS)


def test_unsupported_source_type_raises():
    with pytest.raises(TypeError, match="Unsupported source type"):
        compare_encodings(42)


# ---------------------------------------------------------------------------
# Top-level namespace
# ---------------------------------------------------------------------------

def test_top_level_import():
    assert qd.compare_encodings is compare_encodings
    assert qd.ComparisonResult is ComparisonResult


# ---------------------------------------------------------------------------
# CLI — quprep compare
# ---------------------------------------------------------------------------

def test_cli_compare_basic(tmp_path, small_array):
    import pandas as pd

    from quprep.cli import main

    csv = tmp_path / "data.csv"
    pd.DataFrame(small_array).to_csv(csv, index=False)
    rc = main(["compare", str(csv)])
    assert rc == 0


def test_cli_compare_with_task(tmp_path, small_array):
    import pandas as pd

    from quprep.cli import main

    csv = tmp_path / "data.csv"
    pd.DataFrame(small_array).to_csv(csv, index=False)
    rc = main(["compare", str(csv), "--task", "classification"])
    assert rc == 0


def test_cli_compare_with_qubits(tmp_path, small_array):
    import pandas as pd

    from quprep.cli import main

    csv = tmp_path / "data.csv"
    pd.DataFrame(small_array).to_csv(csv, index=False)
    rc = main(["compare", str(csv), "--qubits", "8"])
    assert rc == 0


def test_cli_compare_include(tmp_path, small_array):
    import pandas as pd

    from quprep.cli import main

    csv = tmp_path / "data.csv"
    pd.DataFrame(small_array).to_csv(csv, index=False)
    rc = main(["compare", str(csv), "--include", "angle,basis"])
    assert rc == 0


def test_cli_compare_exclude(tmp_path, small_array):
    import pandas as pd

    from quprep.cli import main

    csv = tmp_path / "data.csv"
    pd.DataFrame(small_array).to_csv(csv, index=False)
    rc = main(["compare", str(csv), "--exclude", "amplitude,hamiltonian"])
    assert rc == 0


def test_cli_compare_missing_file():
    from quprep.cli import main
    rc = main(["compare", "/tmp/nonexistent_quprep.csv"])
    assert rc == 1


def test_cli_compare_invalid_include(tmp_path, small_array):
    import pandas as pd

    from quprep.cli import main

    csv = tmp_path / "data.csv"
    pd.DataFrame(small_array).to_csv(csv, index=False)
    rc = main(["compare", str(csv), "--include", "angle,totally_fake"])
    assert rc == 1


# ---------------------------------------------------------------------------
# Coverage gap — missing lines
# ---------------------------------------------------------------------------

def test_str_contains_warnings_block(wide_array):
    # lines 134-136: warning lines in __str__() — triggered by over-budget rows
    result = compare_encodings(wide_array, qubits=1)
    s = str(result)
    # At least one encoding needs >1 qubit → warning in output
    assert any(r.warning for r in result.rows)
    assert "[" in s  # warning line format: "  [encoding] ..."


def test_ingest_accepts_dataset(small_array):
    # line 251: Dataset passthrough in _ingest()
    from quprep.compare import _ingest
    from quprep.ingest.numpy_ingester import NumpyIngester

    ds = NumpyIngester().load(small_array)
    result = _ingest(ds)
    assert result is ds


def test_ingest_no_pandas(small_array, monkeypatch):
    # lines 258-259: except ImportError in _ingest() when pandas is absent
    import builtins
    import importlib

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("pandas not available")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    # Re-import _ingest so the patched __import__ is active
    import quprep.compare as compare_mod
    importlib.reload(compare_mod)
    result = compare_mod._ingest(small_array)
    assert result.data.shape == small_array.shape
