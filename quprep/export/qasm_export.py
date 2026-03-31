"""Export encoded data as OpenQASM 3.0 strings or files.

OpenQASM 3.0 is a universal interchange format accepted by all major
quantum frameworks and hardware platforms. Use this exporter for maximum
portability when you don't need framework-specific features.

Supported encodings
-------------------
- angle           : one rotation gate (ry/rx/rz) per qubit.
- entangled_angle : rotation layer + CNOT entangling layer, repeated layers times.
- basis           : X gates on qubits where the bit is 1.
- iqp             : Hadamards + Rz(x_i) + ZZ(x_i·x_j) interactions, repeated reps times.
- reupload        : rotation gate repeated `layers` times per qubit.
- hamiltonian     : Rz(2·x_i·T/S) per qubit, repeated trotter_steps times.
- amplitude       : not supported — exponential-depth state preparation
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
        """
        Convert an EncodedResult to an OpenQASM 3.0 string.

        Parameters
        ----------
        encoded : EncodedResult
            Output from any QuPrep encoder (except amplitude).

        Returns
        -------
        str
            OpenQASM 3.0 circuit string, compatible with Qiskit, Cirq,
            and hardware backends.
        """
        encoding = encoded.metadata.get("encoding", "unknown")
        if encoding == "angle":
            return self._export_angle(encoded)
        if encoding == "entangled_angle":
            return self._export_entangled_angle(encoded)
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
            "Supported: angle, entangled_angle, basis, iqp, reupload, hamiltonian."
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

    def _export_entangled_angle(self, encoded) -> str:
        rotation = encoded.metadata.get("rotation", "ry")
        layers = encoded.metadata.get("layers", 1)
        cnot_pairs = encoded.metadata.get("cnot_pairs", [])
        n = encoded.metadata["n_qubits"]
        x = encoded.parameters
        lines = [
            "OPENQASM 3.0;",
            'include "stdgates.inc";',
            f"qubit[{n}] q;",
        ]
        for _ in range(layers):
            for i, angle in enumerate(x):
                lines.append(f"{rotation}({float(angle)}) q[{i}];")
            for ctrl, tgt in cnot_pairs:
                lines.append(f"cx q[{ctrl}], q[{tgt}];")
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
        """
        Export a list of EncodedResults to QASM strings.

        Parameters
        ----------
        encoded_list : list of EncodedResult

        Returns
        -------
        list of str
            One OpenQASM 3.0 string per sample.
        """
        return [self.export(e) for e in encoded_list]

    def save(self, encoded, path: str | Path) -> None:
        """
        Export an EncodedResult and write the QASM string to a file.

        Parameters
        ----------
        encoded : EncodedResult
            Output from any QuPrep encoder.
        path : str or Path
            Destination file path (e.g. ``'circuit.qasm'``).
        """
        qasm_str = self.export(encoded)
        Path(path).write_text(qasm_str, encoding="utf-8")

    def save_batch(
        self,
        encoded_list: list,
        directory: str | Path,
        stem: str = "circuit",
    ) -> list[Path]:
        """
        Export a batch of EncodedResults to individual QASM files.

        Files are named ``{stem}_0000.qasm``, ``{stem}_0001.qasm``, etc.
        The output directory is created automatically if it does not exist.

        Parameters
        ----------
        encoded_list : list of EncodedResult
            Outputs from any QuPrep encoder.
        directory : str or Path
            Output directory.
        stem : str
            Filename stem (default: ``'circuit'``).

        Returns
        -------
        list of Path
            Paths of the written files, in sample order.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, encoded in enumerate(encoded_list):
            path = directory / f"{stem}_{i:04d}.qasm"
            self.save(encoded, path)
            paths.append(path)
        return paths
