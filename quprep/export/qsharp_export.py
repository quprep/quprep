"""Export encoded data as Q# (Microsoft Quantum) program strings.

Generates standalone Q# 1.0 programs that can be submitted to the
Azure Quantum service or run locally via the ``qsharp`` Python package.

The output is a Q# source string containing an ``operation`` that
applies the encoded feature map to a qubit register.

Supported encodings
-------------------
- angle           : Ry/Rx/Rz gate per qubit.
- entangled_angle : rotation layer + CNOT entangling layer, repeated layers times.
- basis           : X gates on qubits where the bit is 1.
- iqp             : H + Rz(x_i) + ZZ(x_i·x_j) interactions, repeated reps times.
- zz_feature_map  : H + Rz single-qubit + CNOT-Rz-CNOT pairwise, repeated reps times.
- reupload        : rotation gate repeated `layers` times per qubit.
- hamiltonian     : Rz(2·x_i·T/S) per qubit, repeated trotter_steps times.
- tensor_product  : Ry + Rz per qubit from alternating parameter pairs.
- amplitude       : not supported.

No additional dependencies required for string generation.
To submit to Azure Quantum: pip install quprep[qsharp]
"""

from __future__ import annotations

_INDENT = "        "  # 8-space indent inside operation body


class QSharpExporter:
    """
    Export EncodedResult objects to Q# 1.0 program strings.

    No additional dependencies required to generate Q# source.
    To run locally install the ``qsharp`` Python package:
    ``pip install quprep[qsharp]``

    Parameters
    ----------
    namespace : str
        Q# namespace for the generated operation. Default ``"QuPrepCircuit"``.
    operation_name : str
        Name of the generated Q# operation. Default ``"Encode"``.
    """

    def __init__(
        self,
        namespace: str = "QuPrepCircuit",
        operation_name: str = "Encode",
    ):
        self.namespace = namespace
        self.operation_name = operation_name

    def export(self, encoded) -> str:
        """
        Convert an EncodedResult to a Q# program string.

        Parameters
        ----------
        encoded : EncodedResult
            Output from any QuPrep encoder (except amplitude).

        Returns
        -------
        str
            Complete Q# 1.0 source file contents.
        """
        encoding = encoded.metadata.get("encoding", "unknown")
        n = encoded.metadata.get("n_qubits", len(encoded.parameters))
        params = encoded.parameters

        body_lines = self._build_body(encoding, n, params, encoded.metadata)

        lines = [
            f"namespace {self.namespace} {{",
            "    open Microsoft.Quantum.Intrinsic;",
            "    open Microsoft.Quantum.Math;",
            "",
            f"    operation {self.operation_name}() : Unit {{",
            f"        use q = Qubit[{n}];",
        ]
        lines.extend(body_lines)
        lines += [
            "        ResetAll(q);",
            "    }",
            "}",
            "",
        ]
        return "\n".join(lines)

    def _build_body(self, encoding: str, n: int, params, meta: dict) -> list[str]:
        lines: list[str] = []

        if encoding == "amplitude":
            raise NotImplementedError(
                "Amplitude encoding is not supported by QSharpExporter."
            )

        if encoding == "angle":
            rotation = meta.get("rotation", "ry")
            gate = {"ry": "Ry", "rx": "Rx", "rz": "Rz"}[rotation]
            for i, angle in enumerate(params):
                lines.append(f"{_INDENT}{gate}({float(angle)}, q[{i}]);")
            return lines

        if encoding == "entangled_angle":
            rotation = meta.get("rotation", "ry")
            layers = meta.get("layers", 1)
            cnot_pairs = meta.get("cnot_pairs", [])
            gate = {"ry": "Ry", "rx": "Rx", "rz": "Rz"}[rotation]
            for _ in range(layers):
                for i, angle in enumerate(params):
                    lines.append(f"{_INDENT}{gate}({float(angle)}, q[{i}]);")
                for ctrl, tgt in cnot_pairs:
                    lines.append(f"{_INDENT}CNOT(q[{ctrl}], q[{tgt}]);")
            return lines

        if encoding == "basis":
            for i, bit in enumerate(params):
                if bit == 1.0:
                    lines.append(f"{_INDENT}X(q[{i}]);")
            return lines

        if encoding == "iqp":
            d = n
            reps = meta.get("reps", 2)
            x = params[:d]
            pair_angles = params[d:]
            for _ in range(reps):
                for i in range(d):
                    lines.append(f"{_INDENT}H(q[{i}]);")
                for i in range(d):
                    lines.append(f"{_INDENT}Rz({float(x[i])}, q[{i}]);")
                idx = 0
                for i in range(d):
                    for j in range(i + 1, d):
                        angle = float(pair_angles[idx])
                        lines.append(f"{_INDENT}CNOT(q[{i}], q[{j}]);")
                        lines.append(f"{_INDENT}Rz({angle}, q[{j}]);")
                        lines.append(f"{_INDENT}CNOT(q[{i}], q[{j}]);")
                        idx += 1
            return lines

        if encoding == "zz_feature_map":
            reps = meta.get("reps", 2)
            single_angles = meta["single_angles"]
            pair_angles = meta["pair_angles"]
            pairs = meta["pairs"]
            for _ in range(reps):
                for i in range(n):
                    lines.append(f"{_INDENT}H(q[{i}]);")
                for i, angle in enumerate(single_angles):
                    lines.append(f"{_INDENT}Rz({float(angle)}, q[{i}]);")
                for (i, j), angle in zip(pairs, pair_angles):
                    lines.append(f"{_INDENT}CNOT(q[{i}], q[{j}]);")
                    lines.append(f"{_INDENT}Rz({float(angle)}, q[{j}]);")
                    lines.append(f"{_INDENT}CNOT(q[{i}], q[{j}]);")
            return lines

        if encoding == "tensor_product":
            ry_angles = meta["ry_angles"]
            rz_angles = meta["rz_angles"]
            for k in range(n):
                lines.append(f"{_INDENT}Ry({float(ry_angles[k])}, q[{k}]);")
                lines.append(f"{_INDENT}Rz({float(rz_angles[k])}, q[{k}]);")
            return lines

        if encoding == "pauli_feature_map":
            reps = meta.get("reps", 2)
            single_terms = meta.get("single_terms", {})
            pair_terms = meta.get("pair_terms", {})
            _gate = {"X": "Rx", "Y": "Ry", "Z": "Rz"}
            for _ in range(reps):
                for i in range(n):
                    lines.append(f"{_INDENT}H(q[{i}]);")
                for pauli, angles in single_terms.items():
                    gate = _gate[pauli]
                    for i, angle in enumerate(angles):
                        lines.append(f"{_INDENT}{gate}({float(angle)}, q[{i}]);")
                for _pauli, entries in pair_terms.items():
                    for i, j, angle in entries:
                        lines.append(f"{_INDENT}CNOT(q[{i}], q[{j}]);")
                        lines.append(f"{_INDENT}Rz({float(angle)}, q[{j}]);")
                        lines.append(f"{_INDENT}CNOT(q[{i}], q[{j}]);")
            return lines

        if encoding == "random_fourier":
            # Output angles are in [0,π] — encode as Ry rotations
            rotation = meta.get("rotation", "ry")
            gate = {"ry": "Ry", "rx": "Rx", "rz": "Rz"}.get(rotation, "Ry")
            for i, angle in enumerate(params):
                lines.append(f"{_INDENT}{gate}({float(angle)}, q[{i}]);")
            return lines

        if encoding == "reupload":
            rotation = meta.get("rotation", "ry")
            layers = meta.get("layers", 3)
            gate = {"ry": "Ry", "rx": "Rx", "rz": "Rz"}[rotation]
            for _ in range(layers):
                for i, angle in enumerate(params):
                    lines.append(f"{_INDENT}{gate}({float(angle)}, q[{i}]);")
            return lines

        if encoding == "hamiltonian":
            trotter_steps = meta.get("trotter_steps", 4)
            for _ in range(trotter_steps):
                for i, angle in enumerate(params):
                    lines.append(f"{_INDENT}Rz({float(angle)}, q[{i}]);")
            return lines

        raise ValueError(
            f"Unknown encoding '{encoding}'. "
            "Supported: angle, entangled_angle, basis, iqp, zz_feature_map, "
            "pauli_feature_map, random_fourier, tensor_product, reupload, hamiltonian."
        )

    def export_batch(self, encoded_list: list) -> list[str]:
        """
        Export a list of EncodedResults to Q# program strings.

        Parameters
        ----------
        encoded_list : list of EncodedResult

        Returns
        -------
        list of str
        """
        return [self.export(e) for e in encoded_list]
