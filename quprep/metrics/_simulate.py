"""Lightweight numpy statevector simulator for circuit quality metrics.

Supports all 12 QuPrep encodings. Limited to _MAX_QUBITS qubits.
"""

from __future__ import annotations

import numpy as np

# 2^14 = 16 384 complex128 values ≈ 256 KB; fast enough for metric sampling.
MAX_QUBITS = 12

_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_X = np.array([[0, 1], [1, 0]], dtype=complex)


def _ry(t: float) -> np.ndarray:
    c, s = np.cos(t / 2.0), np.sin(t / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rx(t: float) -> np.ndarray:
    c, s = np.cos(t / 2.0), np.sin(t / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _rz(t: float) -> np.ndarray:
    return np.array(
        [[np.exp(-1j * t / 2.0), 0.0], [0.0, np.exp(1j * t / 2.0)]], dtype=complex
    )


class Statevector:
    """Mutable n-qubit statevector.  Qubit 0 = most-significant (leftmost) bit."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.state = np.zeros(1 << n, dtype=complex)
        self.state[0] = 1.0

    # ------------------------------------------------------------------
    # Core gate application
    # ------------------------------------------------------------------

    def apply_single(self, gate: np.ndarray, qubit: int) -> None:
        """Apply a 2×2 gate to `qubit`."""
        s = self.state.reshape([2] * self.n)
        s = np.tensordot(gate, s, axes=([1], [qubit]))
        self.state = np.moveaxis(s, 0, qubit).reshape(-1)

    def apply_cnot(self, ctrl: int, tgt: int) -> None:
        """CNOT: flip `tgt` when `ctrl` = 1."""
        n = self.n
        s = self.state.reshape([2] * n)
        idx: list = [slice(None)] * n
        idx[ctrl] = 1
        idx = tuple(idx)
        tgt_sub = tgt if tgt < ctrl else tgt - 1
        sub = s[idx].copy()
        s[idx] = np.flip(sub, axis=tgt_sub)
        self.state = s.reshape(-1)

    def apply_ising_zz(self, theta: float, q0: int, q1: int) -> None:
        """IsingZZ(θ) = exp(−i·θ/2·Z⊗Z)."""
        s = self.state.reshape([2] * self.n)
        for b0 in range(2):
            for b1 in range(2):
                z0, z1 = 1 - 2 * b0, 1 - 2 * b1
                phase = np.exp(-1j * theta / 2.0 * z0 * z1)
                idx: list = [slice(None)] * self.n
                idx[q0] = b0
                idx[q1] = b1
                s[tuple(idx)] *= phase
        self.state = s.reshape(-1)

    # ------------------------------------------------------------------
    # Named-gate helpers
    # ------------------------------------------------------------------

    def h(self, q: int) -> None:
        self.apply_single(_H, q)

    def ry(self, t: float, q: int) -> None:
        self.apply_single(_ry(t), q)

    def rx(self, t: float, q: int) -> None:
        self.apply_single(_rx(t), q)

    def rz(self, t: float, q: int) -> None:
        self.apply_single(_rz(t), q)

    def x(self, q: int) -> None:
        self.apply_single(_X, q)

    def cnot(self, ctrl: int, tgt: int) -> None:
        self.apply_cnot(ctrl, tgt)

    def ising_zz(self, theta: float, q0: int, q1: int) -> None:
        self.apply_ising_zz(theta, q0, q1)


# ---------------------------------------------------------------------------
# Statevector from EncodedResult
# ---------------------------------------------------------------------------

def statevector_from_encoded(encoded) -> np.ndarray | None:
    """
    Compute the statevector for one EncodedResult using numpy simulation.

    Returns ``None`` if the encoding is unsupported or ``n_qubits > MAX_QUBITS``.
    """
    meta = encoded.metadata
    encoding: str = meta.get("encoding", "")
    n: int = meta.get("n_qubits", 0)

    if n <= 0 or n > MAX_QUBITS:
        return None

    params = np.asarray(encoded.parameters, dtype=float)
    sv = Statevector(n)

    if encoding == "amplitude":
        arr = np.asarray(params, dtype=complex)
        norm = np.linalg.norm(arr)
        sv.state = arr / norm if norm > 1e-12 else arr
        return sv.state

    if encoding == "basis":
        idx = int(sum(int(b) * (1 << (n - 1 - i)) for i, b in enumerate(params)))
        sv.state = np.zeros(1 << n, dtype=complex)
        sv.state[idx % (1 << n)] = 1.0
        return sv.state

    if encoding == "angle":
        _gate = {"ry": sv.ry, "rx": sv.rx, "rz": sv.rz}.get(meta.get("rotation", "ry"))
        if _gate is None:
            return None
        for i, angle in enumerate(params):
            _gate(float(angle), i)
        return sv.state

    if encoding in ("reupload", "random_fourier"):
        _gate = {"ry": sv.ry, "rx": sv.rx, "rz": sv.rz}.get(
            meta.get("rotation", "ry"), sv.ry
        )
        layers = meta.get("layers", 1) if encoding == "reupload" else 1
        for _ in range(layers):
            for i, angle in enumerate(params[:n]):
                _gate(float(angle), i)
        return sv.state

    if encoding == "tensor_product":
        ry_a = meta.get("ry_angles", [])
        rz_a = meta.get("rz_angles", [])
        for k in range(n):
            sv.ry(float(ry_a[k]), k)
            sv.rz(float(rz_a[k]), k)
        return sv.state

    if encoding == "hamiltonian":
        steps = meta.get("trotter_steps", 4)
        for _ in range(steps):
            for i, angle in enumerate(params):
                sv.rz(float(angle), i)
        return sv.state

    if encoding == "entangled_angle":
        _gate = {"ry": sv.ry, "rx": sv.rx, "rz": sv.rz}.get(
            meta.get("rotation", "ry"), sv.ry
        )
        layers = meta.get("layers", 1)
        cnot_pairs = meta.get("cnot_pairs", [])
        for _ in range(layers):
            for i, angle in enumerate(params):
                _gate(float(angle), i)
            for ctrl, tgt in cnot_pairs:
                sv.cnot(ctrl, tgt)
        return sv.state

    if encoding == "iqp":
        reps = meta.get("reps", 2)
        x = params[:n]
        pair_angles = params[n:]
        for _ in range(reps):
            for i in range(n):
                sv.h(i)
            for i in range(n):
                # Havlíček 2019: exp(i·x_i·Z_i); Rz(θ)=exp(-i·θ/2·Z) → θ = -2·x_i
                sv.rz(-2.0 * float(x[i]), i)
            k = 0
            for i in range(n):
                for j in range(i + 1, n):
                    # exp(i·x_i·x_j·Z⊗Z); IsingZZ(θ)=exp(-i·θ/2·Z⊗Z) → θ = -2·x_i·x_j
                    sv.ising_zz(-2.0 * float(pair_angles[k]), i, j)
                    k += 1
        return sv.state

    if encoding == "zz_feature_map":
        reps = meta.get("reps", 1)
        s_angles = meta.get("single_angles", [])
        p_angles = meta.get("pair_angles", [])
        pairs = meta.get("pairs", [])
        for _ in range(reps):
            for i in range(n):
                sv.h(i)
            for i, angle in enumerate(s_angles):
                sv.rz(float(angle), i)
            for (i, j), angle in zip(pairs, p_angles):
                sv.cnot(i, j)
                sv.rz(float(angle), j)
                sv.cnot(i, j)
        return sv.state

    if encoding == "pauli_feature_map":
        reps = meta.get("reps", 2)
        single_terms = meta.get("single_terms", {})
        pair_terms = meta.get("pair_terms", {})
        _pl = {"X": sv.rx, "Y": sv.ry, "Z": sv.rz}
        for _ in range(reps):
            for i in range(n):
                sv.h(i)
            for pauli, angles in single_terms.items():
                gfn = _pl.get(pauli, sv.rz)
                for i, angle in enumerate(angles):
                    gfn(float(angle), i)
            for pauli, entries in pair_terms.items():
                for i, j, angle in entries:
                    # Basis-change rotations to ZZ frame before CNOT-Rz-CNOT block.
                    # X basis: H; Y basis: Rz(-π/2) (S†, maps Y→Z up to global phase)
                    if pauli == "XX":
                        sv.h(i)
                        sv.h(j)
                    elif pauli == "YY":
                        sv.rz(-np.pi / 2.0, i)
                        sv.rz(-np.pi / 2.0, j)
                    elif pauli == "XZ":
                        sv.h(i)
                    elif pauli == "ZX":
                        sv.h(j)
                    elif pauli == "XY":
                        sv.h(i)
                        sv.rz(-np.pi / 2.0, j)
                    elif pauli == "YX":
                        sv.rz(-np.pi / 2.0, i)
                        sv.h(j)
                    elif pauli == "YZ":
                        sv.rz(-np.pi / 2.0, i)
                    elif pauli == "ZY":
                        sv.rz(-np.pi / 2.0, j)
                    sv.cnot(i, j)
                    sv.rz(float(angle), j)
                    sv.cnot(i, j)
                    # Undo basis-change rotations.
                    if pauli == "XX":
                        sv.h(i)
                        sv.h(j)
                    elif pauli == "YY":
                        sv.rz(np.pi / 2.0, i)
                        sv.rz(np.pi / 2.0, j)
                    elif pauli == "XZ":
                        sv.h(i)
                    elif pauli == "ZX":
                        sv.h(j)
                    elif pauli == "XY":
                        sv.h(i)
                        sv.rz(np.pi / 2.0, j)
                    elif pauli == "YX":
                        sv.rz(np.pi / 2.0, i)
                        sv.h(j)
                    elif pauli == "YZ":
                        sv.rz(np.pi / 2.0, i)
                    elif pauli == "ZY":
                        sv.rz(np.pi / 2.0, j)
        return sv.state

    if encoding == "qaoa_problem":
        p = meta.get("p", 1)
        beta = float(meta.get("beta", np.pi / 8.0))
        local_a = meta.get("local_angles", [])
        coupling_a = meta.get("coupling_angles", [])
        pairs = meta.get("pairs", [])
        for i in range(n):
            sv.h(i)
        for _ in range(p):
            for i in range(n):
                sv.rz(2.0 * float(local_a[i]), i)
            for k, (i, j) in enumerate(pairs):
                sv.cnot(i, j)
                sv.rz(2.0 * float(coupling_a[k]), j)
                sv.cnot(i, j)
            for i in range(n):
                sv.rx(2.0 * beta, i)
        return sv.state

    return None  # unsupported encoding
