"""Export encoded data as OpenQASM 3.0 strings or files.

OpenQASM 3.0 is a universal interchange format accepted by all major
quantum frameworks and hardware platforms. Use this exporter for maximum
portability when you don't need framework-specific features.

Supported encodings
-------------------
- angle       : one rotation gate (ry/rx/rz) per qubit.
- basis       : X gates on qubits where the bit is 1.
- iqp         : Hadamards + Rz(x_i) + ZZ(x_i·x_j) interactions, repeated reps times.
- reupload    : rotation gate repeated `layers` times per qubit.
- hamiltonian : Rz(2·x_i·T/S) per qubit, repeated trotter_steps times.
- amplitude   : not supported — exponential-depth state preparation
                has no simple QASM representation. Use QiskitExporter instead.
"""

from __future__ import annotations

from pathlib import Path


class QASMExporter:
    """
    Export EncodedResult objects to OpenQASM 3.0.

    No additional dependencies required.

    Parameters
    ----------
    version : str
        QASM version. Currently only '3.0' is supported.
    """

    def __init__(self, version: str = "3.0"):
        if version != "3.0":
            raise ValueError("Only OpenQASM 3.0 is supported.")
        self.version = version

    def export(self, encoded) -> str:
        """Return the OpenQASM 3.0 string for an EncodedResult."""
        encoding = encoded.metadata.get("encoding", "unknown")
        if encoding == "angle":
            return self._export_angle(encoded)
        if encoding == "basis":
            return self._export_basis(encoded)
        if encoding == "iqp":
            return self._export_iqp(encoded)
        if encoding == "reupload":
            return self._export_reupload(encoded)
        if encoding == "hamiltonian":
            return self._export_hamiltonian(encoded)
        if encoding == "amplitude":
            raise NotImplementedError(
                "Amplitude encoding requires exponential-depth state preparation "
                "which cannot be expressed as a simple QASM snippet. "
                "Use QiskitExporter for amplitude encoding."
            )
        raise ValueError(
            f"Unknown encoding '{encoding}'. "
            "Supported: angle, basis, iqp, reupload, hamiltonian."
        )

    def _export_angle(self, encoded) -> str:
        rotation = encoded.metadata.get("rotation", "ry")
        n = len(encoded.parameters)
        lines = [
            "OPENQASM 3.0;",
            'include "stdgates.inc";',
            f"qubit[{n}] q;",
        ]
        for i, angle in enumerate(encoded.parameters):
            lines.append(f"{rotation}({float(angle)}) q[{i}];")
        return "\n".join(lines) + "\n"

    def _export_basis(self, encoded) -> str:
        n = len(encoded.parameters)
        lines = [
            "OPENQASM 3.0;",
            'include "stdgates.inc";',
            f"qubit[{n}] q;",
        ]
        for i, bit in enumerate(encoded.parameters):
            if bit == 1.0:
                lines.append(f"x q[{i}];")
        return "\n".join(lines) + "\n"

    def _export_iqp(self, encoded) -> str:
        d = encoded.metadata["n_qubits"]
        reps = encoded.metadata.get("reps", 2)
        x = encoded.parameters[:d]
        pairs = encoded.parameters[d:]
        lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', f"qubit[{d}] q;"]
        for _ in range(reps):
            # Hadamard layer
            for i in range(d):
                lines.append(f"h q[{i}];")
            # Single-qubit Z phase
            for i in range(d):
                lines.append(f"rz({float(x[i])}) q[{i}];")
            # ZZ interactions via CX decomposition
            pair_idx = 0
            for i in range(d):
                for j in range(i + 1, d):
                    angle = float(pairs[pair_idx])
                    lines.append(f"cx q[{i}], q[{j}];")
                    lines.append(f"rz({angle}) q[{j}];")
                    lines.append(f"cx q[{i}], q[{j}];")
                    pair_idx += 1
        return "\n".join(lines) + "\n"

    def _export_reupload(self, encoded) -> str:
        d = encoded.metadata["n_qubits"]
        layers = encoded.metadata.get("layers", 3)
        rotation = encoded.metadata.get("rotation", "ry")
        x = encoded.parameters
        lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', f"qubit[{d}] q;"]
        for _ in range(layers):
            for i in range(d):
                lines.append(f"{rotation}({float(x[i])}) q[{i}];")
        return "\n".join(lines) + "\n"

    def _export_hamiltonian(self, encoded) -> str:
        d = encoded.metadata["n_qubits"]
        trotter_steps = encoded.metadata.get("trotter_steps", 4)
        angles = encoded.parameters  # per-step Rz angles
        lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', f"qubit[{d}] q;"]
        for _ in range(trotter_steps):
            for i in range(d):
                lines.append(f"rz({float(angles[i])}) q[{i}];")
        return "\n".join(lines) + "\n"

    def export_batch(self, encoded_list: list) -> list[str]:
        """Export a list of EncodedResults to QASM strings."""
        return [self.export(e) for e in encoded_list]

    def save(self, encoded, path: str | Path) -> None:
        """Write the QASM string to a .qasm file."""
        qasm_str = self.export(encoded)
        Path(path).write_text(qasm_str, encoding="utf-8")
