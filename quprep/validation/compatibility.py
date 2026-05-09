"""Encoding compatibility checks and post-encode invariant verification."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CompatibilityReport:
    """
    Result of :func:`check_compatibility`.

    Attributes
    ----------
    is_compatible : bool
        ``True`` if no hard errors were found.
    errors : list[str]
        Hard failures — encoding will fail or produce wrong results.
    warnings : list[str]
        Soft issues — encoding will run but results may be suboptimal.
    """

    is_compatible: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"CompatibilityReport: {'OK' if self.is_compatible else 'FAIL'}"]
        for e in self.errors:
            lines.append(f"  ERROR   : {e}")
        for w in self.warnings:
            lines.append(f"  WARNING : {w}")
        if not self.errors and not self.warnings:
            lines.append("  No issues found.")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"CompatibilityReport(is_compatible={self.is_compatible}, "
            f"errors={len(self.errors)}, warnings={len(self.warnings)})"
        )


@dataclass
class VerificationReport:
    """
    Result of :func:`verify_encoding`.

    Attributes
    ----------
    passed : bool
        ``True`` if all invariant checks passed.
    checks : list[dict]
        One dict per check: ``{'name', 'passed', 'detail'}``.
    """

    passed: bool
    checks: list[dict] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"VerificationReport: {'PASS' if self.passed else 'FAIL'}"]
        for c in self.checks:
            mark = "OK" if c["passed"] else "FAIL"
            lines.append(f"  [{mark}] {c['name']}: {c['detail']}")
        if not self.checks:
            lines.append("  No invariants checked for this encoder.")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"VerificationReport(passed={self.passed}, checks={len(self.checks)})"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_compatibility(encoder, dataset) -> CompatibilityReport:
    """
    Check a dataset for compatibility with an encoder before encoding runs.

    Catches hard failures (NaNs) and soft issues (wrong value ranges,
    missing fit, padding side-effects) upfront so users get a clear
    message rather than a cryptic error inside the encoder.

    Parameters
    ----------
    encoder : BaseEncoder
        A configured encoder instance.
    dataset : Dataset
        Dataset to check.

    Returns
    -------
    CompatibilityReport
        ``is_compatible`` is ``False`` if any hard errors were found.
    """
    errors: list[str] = []
    warnings: list[str] = []
    data = dataset.data

    # ── NaN check (hard error for all encoders) ───────────────────────────────
    nan_count = int(np.isnan(data).sum())
    if nan_count > 0:
        nan_cols = int(np.isnan(data).any(axis=0).sum())
        errors.append(
            f"{nan_count} NaN value(s) across {nan_cols} feature(s) — "
            "add an Imputer before encoding"
        )

    valid = data[~np.isnan(data)]

    from quprep.encode.amplitude import AmplitudeEncoder
    from quprep.encode.angle import AngleEncoder
    from quprep.encode.basis import BasisEncoder
    from quprep.encode.entangled_angle import EntangledAngleEncoder
    from quprep.encode.hamiltonian import HamiltonianEncoder
    from quprep.encode.iqp import IQPEncoder
    from quprep.encode.pauli_feature_map import PauliFeatureMapEncoder
    from quprep.encode.random_fourier import RandomFourierEncoder
    from quprep.encode.reupload import ReUploadEncoder
    from quprep.encode.tensor_product import TensorProductEncoder
    from quprep.encode.zz_feature_map import ZZFeatureMapEncoder

    d = dataset.n_features

    if isinstance(encoder, AmplitudeEncoder):
        n_qubits = max(1, math.ceil(math.log2(max(d, 2))))
        padded = 2 ** n_qubits
        if padded != d:
            warnings.append(
                f"n_features={d} is not a power of 2 — "
                f"will be zero-padded to {padded} before normalisation"
            )

    elif isinstance(encoder, (AngleEncoder, ReUploadEncoder, EntangledAngleEncoder)):
        rotation = getattr(encoder, "rotation", "ry")
        if rotation == "ry":
            lo, hi, label, scaler = 0.0, float(np.pi), "[0, π]", "minmax_pi"
        else:
            lo, hi, label, scaler = (
                -float(np.pi), float(np.pi), "[-π, π]", "minmax_pm_pi"
            )
        if valid.size > 0 and (float(valid.min()) < lo - 1e-6 or float(valid.max()) > hi + 1e-6):
            warnings.append(
                f"{type(encoder).__name__}(rotation='{rotation}') expects values in "
                f"{label} — add Scaler('{scaler}') to the pipeline"
            )

    elif isinstance(encoder, TensorProductEncoder):
        if valid.size > 0 and (
            float(valid.min()) < -1e-6 or float(valid.max()) > float(np.pi) + 1e-6
        ):
            warnings.append(
                "TensorProductEncoder expects values in [0, π] — "
                "add Scaler('minmax_pi')"
            )

    elif isinstance(encoder, BasisEncoder):
        if valid.size > 0:
            unique_vals = set(np.unique(valid).tolist())
            if not unique_vals.issubset({0.0, 1.0}):
                warnings.append(
                    "BasisEncoder expects binary values in {0, 1} — "
                    "add Scaler('binary')"
                )

    elif isinstance(encoder, (IQPEncoder, PauliFeatureMapEncoder)):
        lo, hi = -float(np.pi), float(np.pi)
        if valid.size > 0 and (float(valid.min()) < lo - 1e-6 or float(valid.max()) > hi + 1e-6):
            warnings.append(
                f"{type(encoder).__name__} expects values in [-π, π] — "
                "add Scaler('minmax_pm_pi')"
            )

    elif isinstance(encoder, ZZFeatureMapEncoder):
        hi = 2.0 * float(np.pi)
        if valid.size > 0 and (float(valid.min()) < -1e-6 or float(valid.max()) > hi + 1e-6):
            warnings.append(
                "ZZFeatureMapEncoder expects values in [0, 2π] — "
                "add Scaler('minmax_2pi')"
            )

    elif isinstance(encoder, RandomFourierEncoder):
        if encoder._W is None:
            warnings.append(
                "RandomFourierEncoder has not been fitted — call encoder.fit(X) "
                "or use Pipeline (which calls fit() automatically)"
            )

    elif isinstance(encoder, HamiltonianEncoder):
        if valid.size > 0 and (float(valid.min()) < -10 or float(valid.max()) > 10):
            warnings.append(
                "HamiltonianEncoder works best with z-score normalised input — "
                "add Scaler('zscore') to the pipeline"
            )

    return CompatibilityReport(
        is_compatible=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def verify_encoding(encoded, encoder) -> VerificationReport:
    """
    Verify post-encoding invariants for a batch of ``EncodedResult`` objects.

    Checks encoding-specific invariants that, if violated, indicate a silent
    pipeline misconfiguration such as a wrong normalizer being applied.

    Parameters
    ----------
    encoded : list[EncodedResult]
        Output of ``encoder.encode_batch()`` or ``PipelineResult.encoded``.
    encoder : BaseEncoder
        The encoder used to produce ``encoded``.

    Returns
    -------
    VerificationReport
    """
    from quprep.encode.amplitude import AmplitudeEncoder
    from quprep.encode.angle import AngleEncoder
    from quprep.encode.basis import BasisEncoder
    from quprep.encode.entangled_angle import EntangledAngleEncoder
    from quprep.encode.iqp import IQPEncoder
    from quprep.encode.pauli_feature_map import PauliFeatureMapEncoder
    from quprep.encode.reupload import ReUploadEncoder
    from quprep.encode.zz_feature_map import ZZFeatureMapEncoder

    checks: list[dict] = []
    if not encoded:
        return VerificationReport(passed=True, checks=[])

    all_params = np.concatenate(
        [np.asarray(e.parameters, dtype=float) for e in encoded]
    )

    if isinstance(encoder, AmplitudeEncoder):
        norms = np.array(
            [np.linalg.norm(np.asarray(e.parameters, dtype=float)) for e in encoded]
        )
        max_err = float(np.max(np.abs(norms - 1.0)))
        checks.append({
            "name": "unit_norm",
            "passed": max_err < 1e-6,
            "detail": f"max |‖ψ‖ − 1| = {max_err:.2e} (threshold 1e-6)",
        })

    elif isinstance(encoder, (AngleEncoder, ReUploadEncoder, EntangledAngleEncoder)):
        rotation = getattr(encoder, "rotation", "ry")
        if rotation == "ry":
            lo, hi, label = 0.0, float(np.pi), "[0, π]"
        else:
            lo, hi, label = -float(np.pi), float(np.pi), "[-π, π]"
        out_count = int(np.sum((all_params < lo - 1e-6) | (all_params > hi + 1e-6)))
        in_range = out_count == 0
        checks.append({
            "name": "angle_range",
            "passed": in_range,
            "detail": (
                f"all values in {label}"
                if in_range
                else f"{out_count} parameter(s) outside {label} — check Scaler strategy"
            ),
        })

    elif isinstance(encoder, BasisEncoder):
        unique_vals = set(np.unique(np.round(all_params, 8)).tolist())
        is_binary = unique_vals.issubset({0.0, 1.0})
        checks.append({
            "name": "binary_values",
            "passed": is_binary,
            "detail": (
                "all values in {0, 1}"
                if is_binary
                else f"unexpected values: {sorted(unique_vals)[:5]}"
            ),
        })

    elif isinstance(encoder, (IQPEncoder, PauliFeatureMapEncoder)):
        lo, hi = -float(np.pi), float(np.pi)
        out_count = int(np.sum((all_params < lo - 1e-6) | (all_params > hi + 1e-6)))
        in_range = out_count == 0
        checks.append({
            "name": "angle_range",
            "passed": in_range,
            "detail": (
                "all values in [-π, π]"
                if in_range
                else f"{out_count} parameter(s) outside [-π, π] — check Scaler strategy"
            ),
        })

    elif isinstance(encoder, ZZFeatureMapEncoder):
        lo, hi = 0.0, 2.0 * float(np.pi)
        out_count = int(np.sum((all_params < lo - 1e-6) | (all_params > hi + 1e-6)))
        in_range = out_count == 0
        checks.append({
            "name": "angle_range",
            "passed": in_range,
            "detail": (
                "all values in [0, 2π]"
                if in_range
                else f"{out_count} parameter(s) outside [0, 2π] — use Scaler('minmax_2pi')"
            ),
        })

    return VerificationReport(
        passed=all(c["passed"] for c in checks) if checks else True,
        checks=checks,
    )
