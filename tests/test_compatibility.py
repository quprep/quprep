"""Tests for check_compatibility and verify_encoding."""

from __future__ import annotations

import numpy as np

import quprep as qd
from quprep.core.dataset import Dataset


def _ds(data, feature_names=None):
    n, d = data.shape
    return Dataset(
        data=data.astype(np.float64),
        feature_names=feature_names or [f"f{i}" for i in range(d)],
        feature_types=["continuous"] * d,
    )


# ---------------------------------------------------------------------------
# check_compatibility
# ---------------------------------------------------------------------------

class TestCheckCompatibility:
    def test_clean_angle_ok(self):
        ds = _ds(np.random.default_rng(0).uniform(0, np.pi, (10, 4)))
        report = qd.check_compatibility(qd.AngleEncoder(), ds)
        assert report.is_compatible
        assert not report.errors
        assert not report.warnings

    def test_nan_is_hard_error(self):
        data = np.ones((5, 3))
        data[0, 1] = np.nan
        ds = _ds(data)
        report = qd.check_compatibility(qd.AngleEncoder(), ds)
        assert not report.is_compatible
        assert len(report.errors) == 1
        assert "NaN" in report.errors[0]

    def test_angle_ry_out_of_range_warns(self):
        ds = _ds(np.random.default_rng(0).uniform(5, 10, (10, 4)))
        report = qd.check_compatibility(qd.AngleEncoder(rotation="ry"), ds)
        assert report.is_compatible
        assert len(report.warnings) == 1
        assert "minmax_pi" in report.warnings[0]

    def test_angle_rx_out_of_range_warns(self):
        ds = _ds(np.random.default_rng(0).uniform(5, 10, (10, 4)))
        report = qd.check_compatibility(qd.AngleEncoder(rotation="rx"), ds)
        assert report.is_compatible
        assert len(report.warnings) == 1
        assert "minmax_pm_pi" in report.warnings[0]

    def test_basis_non_binary_warns(self):
        ds = _ds(np.random.default_rng(0).uniform(0, 5, (10, 4)))
        report = qd.check_compatibility(qd.BasisEncoder(), ds)
        assert report.is_compatible
        assert any("binary" in w for w in report.warnings)

    def test_basis_binary_ok(self):
        data = np.random.default_rng(0).choice([0.0, 1.0], size=(10, 4))
        ds = _ds(data)
        report = qd.check_compatibility(qd.BasisEncoder(), ds)
        assert report.is_compatible
        assert not report.warnings

    def test_amplitude_non_power_of_two_warns(self):
        ds = _ds(np.ones((5, 5)))
        report = qd.check_compatibility(qd.AmplitudeEncoder(), ds)
        assert report.is_compatible
        assert any("zero-padded" in w for w in report.warnings)

    def test_amplitude_power_of_two_ok(self):
        ds = _ds(np.ones((5, 4)))
        report = qd.check_compatibility(qd.AmplitudeEncoder(), ds)
        assert report.is_compatible
        assert not report.warnings

    def test_zz_out_of_range_warns(self):
        ds = _ds(np.random.default_rng(0).uniform(5, 10, (10, 3)))
        report = qd.check_compatibility(qd.ZZFeatureMapEncoder(), ds)
        assert report.is_compatible
        assert any("minmax_2pi" in w for w in report.warnings)

    def test_str_repr(self):
        ds = _ds(np.ones((5, 3)))
        report = qd.check_compatibility(qd.AngleEncoder(), ds)
        assert "CompatibilityReport" in str(report)
        assert "CompatibilityReport" in repr(report)


# ---------------------------------------------------------------------------
# verify_encoding
# ---------------------------------------------------------------------------

class TestVerifyEncoding:
    def test_amplitude_unit_norm_pass(self):
        rng = np.random.default_rng(0)
        data = rng.uniform(0, 1, (5, 4))
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        data = data / norms
        ds = _ds(data)
        enc = qd.AmplitudeEncoder()
        encoded = enc.encode_batch(ds)
        report = qd.verify_encoding(encoded, enc)
        assert report.passed
        assert report.checks[0]["name"] == "unit_norm"

    def test_angle_in_range_pass(self):
        data = np.random.default_rng(0).uniform(0, np.pi, (5, 4))
        ds = _ds(data)
        enc = qd.AngleEncoder(rotation="ry")
        encoded = enc.encode_batch(ds)
        report = qd.verify_encoding(encoded, enc)
        assert report.passed

    def test_angle_out_of_range_fail(self):
        data = np.random.default_rng(0).uniform(5, 10, (5, 4))
        ds = _ds(data)
        enc = qd.AngleEncoder(rotation="ry")
        encoded = enc.encode_batch(ds)
        report = qd.verify_encoding(encoded, enc)
        assert not report.passed
        assert "outside" in report.checks[0]["detail"]

    def test_empty_encoded(self):
        report = qd.verify_encoding([], qd.AngleEncoder())
        assert report.passed
        assert report.checks == []

    def test_basis_binary_pass(self):
        data = np.random.default_rng(0).choice([0.0, 1.0], size=(5, 4)).astype(float)
        ds = _ds(data)
        enc = qd.BasisEncoder()
        encoded = enc.encode_batch(ds)
        report = qd.verify_encoding(encoded, enc)
        assert report.passed

    def test_str_repr(self):
        report = qd.verify_encoding([], qd.AngleEncoder())
        assert "VerificationReport" in str(report)
        assert "VerificationReport" in repr(report)
