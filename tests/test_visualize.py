"""Tests for draw_ascii() and draw_matplotlib()."""

from __future__ import annotations

import numpy as np
import pytest

from quprep.encode.amplitude import AmplitudeEncoder
from quprep.encode.angle import AngleEncoder
from quprep.encode.basis import BasisEncoder
from quprep.encode.entangled_angle import EntangledAngleEncoder
from quprep.encode.hamiltonian import HamiltonianEncoder
from quprep.encode.iqp import IQPEncoder
from quprep.encode.reupload import ReUploadEncoder
from quprep.export.visualize import draw_ascii, draw_matplotlib

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

X3 = np.array([0.5, 1.2, 0.75])
X4 = np.array([0.1, 0.9, 0.4, 0.8])


def _enc(encoder, x=X3):
    return encoder.encode(x)


# ---------------------------------------------------------------------------
# draw_ascii — return type and basic structure
# ---------------------------------------------------------------------------


class TestDrawAscii:
    def test_returns_string(self):
        result = draw_ascii(_enc(AngleEncoder()))
        assert isinstance(result, str)

    def test_contains_qubit_labels(self):
        result = draw_ascii(_enc(AngleEncoder()))
        assert "q[0]" in result
        assert "q[1]" in result
        assert "q[2]" in result

    def test_n_lines_correct(self):
        # header + 2n-1 qubit/gap rows + trailing newline → n+1 non-empty lines
        enc = _enc(AngleEncoder())
        n = enc.metadata["n_qubits"]
        lines = draw_ascii(enc).strip().splitlines()
        # header line + 2n-1 data rows = 2n rows total
        assert len(lines) == 2 * n

    def test_header_contains_encoding_name(self):
        result = draw_ascii(_enc(AngleEncoder()))
        assert "angle" in result

    def test_header_contains_qubit_count(self):
        result = draw_ascii(_enc(AngleEncoder()))
        assert "3 qubits" in result

    # ------------------------------------------------------------------
    # Angle encoding
    # ------------------------------------------------------------------

    def test_angle_ry_label(self):
        result = draw_ascii(_enc(AngleEncoder(rotation="ry")))
        assert "RY(" in result

    def test_angle_rx_label(self):
        result = draw_ascii(_enc(AngleEncoder(rotation="rx")))
        assert "RX(" in result

    def test_angle_rz_label(self):
        result = draw_ascii(_enc(AngleEncoder(rotation="rz")))
        assert "RZ(" in result

    # ------------------------------------------------------------------
    # Basis encoding
    # ------------------------------------------------------------------

    def test_basis_x_gate_present(self):
        x = np.array([1.0, 0.0, 1.0])
        result = draw_ascii(_enc(BasisEncoder(), x))
        assert "X" in result

    def test_basis_all_zero_no_x(self):
        x = np.array([0.0, 0.0, 0.0])
        result = draw_ascii(_enc(BasisEncoder(), x))
        # No X gate labels (only wires and qubit labels)
        lines = result.splitlines()
        wire_lines = [ln for ln in lines if ln.startswith("q[")]
        for line in wire_lines:  # noqa: E741
            assert "[X]" not in line

    # ------------------------------------------------------------------
    # Amplitude encoding
    # ------------------------------------------------------------------

    def test_amplitude_sp_label(self):
        x = np.array([0.5, 0.5, 0.5, 0.5])
        result = draw_ascii(_enc(AmplitudeEncoder(), x / np.linalg.norm(x)))
        assert "SP" in result

    # ------------------------------------------------------------------
    # Entangled angle encoding
    # ------------------------------------------------------------------

    def test_entangled_angle_cnot_symbol(self):
        result = draw_ascii(_enc(EntangledAngleEncoder(layers=1)))
        assert "●" in result
        assert "⊕" in result

    def test_entangled_angle_circular_has_more_cnots(self):
        linear = draw_ascii(_enc(EntangledAngleEncoder(entanglement="linear")))
        circular = draw_ascii(_enc(EntangledAngleEncoder(entanglement="circular")))
        assert circular.count("●") > linear.count("●")

    def test_entangled_angle_full_has_most_cnots(self):
        linear = draw_ascii(_enc(EntangledAngleEncoder(entanglement="linear")))
        full = draw_ascii(_enc(EntangledAngleEncoder(entanglement="full")))
        assert full.count("●") > linear.count("●")

    def test_entangled_angle_two_layers(self):
        enc = _enc(EntangledAngleEncoder(layers=2))
        result = draw_ascii(enc)
        # Each layer adds rotation gates → more gate boxes
        enc1 = _enc(EntangledAngleEncoder(layers=1))
        assert result.count("RY(") > draw_ascii(enc1).count("RY(")

    # ------------------------------------------------------------------
    # IQP encoding
    # ------------------------------------------------------------------

    def test_iqp_h_gate_present(self):
        result = draw_ascii(_enc(IQPEncoder(reps=1)))
        assert "H" in result

    def test_iqp_zz_symbol(self):
        result = draw_ascii(_enc(IQPEncoder(reps=1)))
        assert "Z" in result

    # ------------------------------------------------------------------
    # Re-upload encoding
    # ------------------------------------------------------------------

    def test_reupload_multi_layer(self):
        enc2 = _enc(ReUploadEncoder(layers=2))
        enc1 = _enc(ReUploadEncoder(layers=1))
        assert draw_ascii(enc2).count("RY(") == 2 * draw_ascii(enc1).count("RY(")

    # ------------------------------------------------------------------
    # Hamiltonian encoding
    # ------------------------------------------------------------------

    def test_hamiltonian_rz_present(self):
        result = draw_ascii(_enc(HamiltonianEncoder(trotter_steps=2)))
        assert "Rz(" in result

    def test_hamiltonian_trotter_steps(self):
        enc2 = _enc(HamiltonianEncoder(trotter_steps=2))
        enc1 = _enc(HamiltonianEncoder(trotter_steps=1))
        assert draw_ascii(enc2).count("Rz(") == 2 * draw_ascii(enc1).count("Rz(")

    # ------------------------------------------------------------------
    # width parameter accepted without error
    # ------------------------------------------------------------------

    def test_width_accepted(self):
        result = draw_ascii(_enc(AngleEncoder()), width=120)
        assert isinstance(result, str)

    # ------------------------------------------------------------------
    # Top-level import
    # ------------------------------------------------------------------

    def test_importable_from_quprep(self):
        from quprep import draw_ascii as da  # noqa: F401


# ---------------------------------------------------------------------------
# draw_matplotlib
# ---------------------------------------------------------------------------


class TestDrawMatplotlib:
    def test_returns_figure(self):
        mpl = pytest.importorskip("matplotlib")  # noqa: F841
        fig = draw_matplotlib(_enc(AngleEncoder()))
        import matplotlib.figure
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_all_encodings_no_error(self):
        mpl = pytest.importorskip("matplotlib")  # noqa: F841
        import matplotlib.pyplot as plt

        x4_norm = X4 / np.linalg.norm(X4)
        cases = [
            (AngleEncoder(), X3),
            (BasisEncoder(), X3),
            (AmplitudeEncoder(), x4_norm),
            (EntangledAngleEncoder(), X3),
            (IQPEncoder(reps=1), X3),
            (ReUploadEncoder(layers=1), X3),
            (HamiltonianEncoder(trotter_steps=1), X3),
        ]
        for enc, x in cases:
            fig = draw_matplotlib(enc.encode(x))
            assert fig is not None
            plt.close(fig)

    def test_saves_to_file(self, tmp_path):
        mpl = pytest.importorskip("matplotlib")  # noqa: F841
        out = tmp_path / "circuit.png"
        result = draw_matplotlib(_enc(AngleEncoder()), filename=out)
        assert result is None
        assert out.exists()
        assert out.stat().st_size > 0

    def test_save_pdf(self, tmp_path):
        mpl = pytest.importorskip("matplotlib")  # noqa: F841
        out = tmp_path / "circuit.pdf"
        draw_matplotlib(_enc(AngleEncoder()), filename=out)
        assert out.exists()

    def test_importable_from_quprep(self):
        from quprep import draw_matplotlib as dm  # noqa: F401

    def test_missing_matplotlib_raises(self, monkeypatch):
        import sys
        # Temporarily hide matplotlib
        monkeypatch.setitem(sys.modules, "matplotlib", None)
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
        with pytest.raises((ImportError, TypeError)):
            draw_matplotlib(_enc(AngleEncoder()))
