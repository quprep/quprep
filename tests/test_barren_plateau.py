"""Tests for quprep.metrics.barren_plateau."""

import numpy as np
import pytest

import quprep as qd
from quprep.core.dataset import Dataset
from quprep.metrics.barren_plateau import (
    BarrenPlateauReport,
    _gradient_variance,
    _risk_level,
    detect_barren_plateau,
)


@pytest.fixture
def small_ds():
    rng = np.random.default_rng(0)
    return Dataset(data=rng.uniform(0, 1, (30, 4)))


@pytest.fixture
def large_ds():
    rng = np.random.default_rng(0)
    return Dataset(data=rng.uniform(0, 1, (30, 12)))


class TestGradientVariance:
    def test_global_exponential_decay(self):
        # Var should halve with each additional qubit
        v4 = _gradient_variance(4, "global")
        v5 = _gradient_variance(5, "global")
        assert pytest.approx(v5, rel=1e-9) == v4 / 2

    def test_local_polynomial_decay(self):
        v4 = _gradient_variance(4, "local")
        v8 = _gradient_variance(8, "local")
        assert v4 > v8

    def test_global_less_than_local_for_large_n(self):
        # For large n, global (exp) < local (poly)
        assert _gradient_variance(15, "global") < _gradient_variance(15, "local")


class TestRiskLevel:
    def test_none_small_variance(self):
        assert _risk_level(0.1) == "none"

    def test_mild(self):
        assert _risk_level(0.01) == "mild"

    def test_high(self):
        assert _risk_level(0.001) == "high"

    def test_severe_small_variance(self):
        assert _risk_level(0.0001) == "severe"

    def test_boundary_none(self):
        # 0.05 is the threshold — strictly above → "none"
        assert _risk_level(0.051) == "none"
        assert _risk_level(0.05) == "mild"  # exactly on boundary falls to next bucket

    def test_boundary_mild(self):
        assert _risk_level(0.051) == "none"
        assert _risk_level(0.006) == "mild"


class TestDetectBarrenPlateau:
    def test_returns_report(self, small_ds):
        report = detect_barren_plateau(qd.AngleEncoder(), small_ds)
        assert isinstance(report, BarrenPlateauReport)

    def test_small_circuit_none_risk(self, small_ds):
        # 4 features → 4 qubits → Var=0.125 → "none"
        report = detect_barren_plateau(qd.AngleEncoder(), small_ds)
        assert report.risk_level == "none"
        assert report.mitigations == []

    def test_large_circuit_higher_risk(self, large_ds):
        # 12 features → 12 qubits → Var=2^(-11) ≈ 0.0005 → "high" or "severe"
        report = detect_barren_plateau(qd.IQPEncoder(), large_ds)
        assert report.risk_level in ("high", "severe")
        assert len(report.mitigations) > 0

    def test_local_cost_lowers_risk(self, large_ds):
        global_report = detect_barren_plateau(qd.IQPEncoder(), large_ds, cost_type="global")
        local_report = detect_barren_plateau(qd.IQPEncoder(), large_ds, cost_type="local")
        # local cost variance is higher → equal or lower risk
        risk_order = ["none", "mild", "high", "severe"]
        assert risk_order.index(local_report.risk_level) <= risk_order.index(
            global_report.risk_level
        )

    def test_invalid_cost_type(self, small_ds):
        with pytest.raises(ValueError, match="cost_type"):
            detect_barren_plateau(qd.AngleEncoder(), small_ds, cost_type="magic")

    def test_n_qubits_matches_encoder(self, small_ds):
        report = detect_barren_plateau(qd.AngleEncoder(), small_ds)
        assert report.n_qubits == 4  # 4 features → 4 qubits for angle encoder

    def test_str_representation(self, small_ds):
        report = detect_barren_plateau(qd.AngleEncoder(), small_ds)
        s = str(report)
        assert "BarrenPlateauReport" in s
        assert "n_qubits" in s
        assert "risk_level" in s

    def test_top_level_import(self, small_ds):
        report = qd.detect_barren_plateau(qd.AngleEncoder(), small_ds)
        assert isinstance(report, qd.BarrenPlateauReport)
