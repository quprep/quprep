"""Tests for quantum encoders.

Property-based tests (via hypothesis) are required for all encoders.
Key invariants:
  - AmplitudeEncoder output must satisfy ‖amplitudes‖₂ = 1.
  - AngleEncoder output must have values in the correct rotation range.
  - All encoders must be deterministic (same input → same output).
"""

import numpy as np
import pytest


class TestAngleEncoder:
    def test_output_shape(self):
        pytest.skip("AngleEncoder not yet implemented")

    def test_deterministic(self):
        pytest.skip("AngleEncoder not yet implemented")

    def test_invalid_rotation_raises(self):
        from quprep.encode.angle import AngleEncoder
        with pytest.raises(ValueError):
            AngleEncoder(rotation="rq")


class TestAmplitudeEncoder:
    def test_unit_norm_invariant(self):
        """Amplitude encoder must always produce unit-norm output."""
        pytest.skip("AmplitudeEncoder not yet implemented")

    def test_non_unit_norm_input_raises(self):
        pytest.skip("AmplitudeEncoder not yet implemented")


class TestBasisEncoder:
    def test_binary_input(self):
        pytest.skip("BasisEncoder not yet implemented")
