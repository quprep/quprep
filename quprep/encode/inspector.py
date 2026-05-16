"""Circuit parameter inspector — structured access to encoding gate parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from quprep.encode.base import EncodedResult

_ROTATION_GATE = {"ry": "Ry", "rx": "Rx", "rz": "Rz"}


@dataclass
class GateParam:
    """A single gate operation with its parameters."""

    gate: str
    qubit: int | None = None
    angle: float | None = None
    control: int | None = None
    amplitudes: np.ndarray | None = None

    def __repr__(self) -> str:
        parts = [self.gate]
        if self.control is not None:
            parts.append(f"q{self.control}→q{self.qubit}")
        elif self.qubit is not None:
            parts.append(f"q{self.qubit}")
        if self.angle is not None:
            parts.append(f"θ={self.angle:.4f}")
        if self.amplitudes is not None:
            parts.append(f"amps[{len(self.amplitudes)}]")
        return f"GateParam({', '.join(parts)})"


@dataclass
class EncodingParams:
    """
    Structured representation of a quantum encoding circuit's parameters.

    Attributes
    ----------
    encoding : str
        Encoding type (e.g., ``'angle'``, ``'iqp'``).
    n_qubits : int
        Number of qubits used.
    depth : int or str or None
        Circuit depth.
    angles : np.ndarray
        Raw rotation angle parameters from the encoder.
    gates : list[GateParam]
        Gate-level description of the circuit.
    metadata : dict
        Full metadata dict from the ``EncodedResult``.
    """

    encoding: str
    n_qubits: int
    depth: int | str | None
    angles: np.ndarray
    gates: list[GateParam] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"EncodingParams(encoding={self.encoding!r}, n_qubits={self.n_qubits}, "
            f"depth={self.depth!r}, n_gates={len(self.gates)})"
        )

    def summary(self) -> str:
        """Return a human-readable multi-line summary of all gates."""
        lines = [
            f"Encoding : {self.encoding}",
            f"Qubits   : {self.n_qubits}",
            f"Depth    : {self.depth}",
            f"Gates    : {len(self.gates)}",
            "",
        ]
        for g in self.gates:
            if g.gate in ("Ry", "Rx", "Rz"):
                lines.append(f"  {g.gate}(q{g.qubit}, θ={g.angle:.4f})")
            elif g.gate == "CNOT":
                lines.append(f"  CNOT(q{g.control}→q{g.qubit})")
            elif g.gate in ("H", "X"):
                lines.append(f"  {g.gate}(q{g.qubit})")
            elif g.gate == "IsingZZ":
                lines.append(f"  IsingZZ(q{g.control}⊗q{g.qubit}, θ={g.angle:.4f})")
            elif g.gate == "Initialize":
                lines.append(f"  Initialize(amps[{len(g.amplitudes)}])")
            else:
                lines.append(f"  {g.gate}")
        return "\n".join(lines)


def inspect_encoding(result: EncodedResult) -> EncodingParams:
    """
    Inspect an ``EncodedResult`` and return structured gate parameters.

    Parses the ``parameters`` and ``metadata`` fields of the result to build
    a gate-level description of the encoded circuit, suitable for programmatic
    inspection and debugging.

    Parameters
    ----------
    result : EncodedResult
        Output of any ``BaseEncoder.encode()`` call.

    Returns
    -------
    EncodingParams
        Structured object with raw ``angles``, per-gate ``GateParam`` list,
        qubit count, depth, and full metadata.

    Examples
    --------
    >>> from quprep.encode.angle import AngleEncoder
    >>> from quprep.encode.inspector import inspect_encoding
    >>> import numpy as np
    >>> result = AngleEncoder().encode(np.array([0.5, 1.0, 1.5]))
    >>> ep = inspect_encoding(result)
    >>> ep.n_qubits
    3
    >>> ep.gates[0]
    GateParam(Ry, q0, θ=0.5000)
    """
    meta = result.metadata
    enc: str = meta.get("encoding", "unknown")
    n_qubits: int = int(meta.get("n_qubits", 0))
    depth = meta.get("depth")
    params = np.asarray(result.parameters, dtype=float)

    gates = _parse_gates(enc, params, meta, n_qubits)

    return EncodingParams(
        encoding=enc,
        n_qubits=n_qubits,
        depth=depth,
        angles=params,
        gates=gates,
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Gate parsers per encoding type
# ---------------------------------------------------------------------------

def _parse_gates(
    enc: str,
    params: np.ndarray,
    meta: dict,
    n: int,
) -> list[GateParam]:
    if enc == "angle":
        gname = _ROTATION_GATE.get(meta.get("rotation", "ry"), "Ry")
        return [GateParam(gate=gname, qubit=i, angle=float(params[i])) for i in range(n)]

    if enc in ("dense_angle", "tensor_product"):
        first_rot = meta.get("first_rotation", "ry")
        r1 = _ROTATION_GATE.get(first_rot, "Ry")
        r2 = _ROTATION_GATE.get(meta.get("second_rotation", "rz"), "Rz")
        if enc == "tensor_product":
            ry_a = meta.get("ry_angles", params[0::2].tolist())
            rz_a = meta.get("rz_angles", params[1::2].tolist())
            r1, r2 = "Ry", "Rz"
        else:
            ry_a = params[0::2].tolist()
            rz_a = params[1::2].tolist()
        gates: list[GateParam] = []
        for k in range(n):
            gates.append(GateParam(gate=r1, qubit=k, angle=float(ry_a[k])))
            gates.append(GateParam(gate=r2, qubit=k, angle=float(rz_a[k])))
        return gates

    if enc == "iqp":
        reps = meta.get("reps", 2)
        x = params[:n]
        pair_angles = params[n:]
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        gates = []
        for _ in range(reps):
            for i in range(n):
                gates.append(GateParam(gate="H", qubit=i))
            for i in range(n):
                gates.append(GateParam(gate="Rz", qubit=i, angle=-2.0 * float(x[i])))
            for k, (i, j) in enumerate(pairs):
                gates.append(GateParam(gate="IsingZZ", qubit=j, control=i,
                                       angle=-2.0 * float(pair_angles[k])))
        return gates

    if enc == "amplitude":
        return [GateParam(gate="Initialize", amplitudes=params)]

    if enc == "basis":
        return [GateParam(gate="X", qubit=i) for i, b in enumerate(params) if b > 0.5]

    if enc == "discretized":
        return [GateParam(gate="X", qubit=i) for i, b in enumerate(params) if b > 0.5]

    if enc == "entangled_angle":
        gname = _ROTATION_GATE.get(meta.get("rotation", "ry"), "Ry")
        layers = meta.get("layers", 1)
        cnot_pairs = meta.get("cnot_pairs", [])
        gates = []
        for _ in range(layers):
            for i in range(n):
                gates.append(GateParam(gate=gname, qubit=i, angle=float(params[i])))
            for ctrl, tgt in cnot_pairs:
                gates.append(GateParam(gate="CNOT", qubit=tgt, control=ctrl))
        return gates

    if enc in ("reupload", "random_fourier"):
        gname = _ROTATION_GATE.get(meta.get("rotation", "ry"), "Ry")
        layers = meta.get("layers", 1) if enc == "reupload" else 1
        gates = []
        for _ in range(layers):
            for i in range(n):
                gates.append(GateParam(gate=gname, qubit=i, angle=float(params[i])))
        return gates

    if enc == "hamiltonian":
        steps = meta.get("trotter_steps", 4)
        gates = []
        for _ in range(steps):
            for i, angle in enumerate(params):
                gates.append(GateParam(gate="Rz", qubit=i, angle=float(angle)))
        return gates

    # Generic fallback — expose raw parameters as Ry rotations
    safe_n = max(n, 1)
    return [
        GateParam(gate="Ry", qubit=i % safe_n, angle=float(p))
        for i, p in enumerate(params)
    ]
