"""Export encoded data as Cirq Circuit objects.

Supported encodings
-------------------
- angle           : Ry/Rx/Rz gate per qubit.
- entangled_angle : rotation layer + CNOT entangling layer, repeated layers times.
- basis           : X gates on qubits where the bit is 1.
- iqp             : H + Rz(x_i) + CX+Rz(x_i·x_j)+CX interactions, repeated reps times.
- reupload        : rotation gate repeated ``layers`` times per qubit.
- hamiltonian     : Rz(2·x_i·T/S) per qubit, repeated trotter_steps times.
- zz_feature_map  : H + Rz single-qubit + CNOT-Rz-CNOT pairwise, repeated reps times.
- pauli_feature_map: Generalised Pauli feature map (X/Y/Z, XX/YY/ZZ strings).
- random_fourier  : angle-encoded Fourier features (treated as angle encoding).
- tensor_product  : Ry + Rz per qubit from alternating parameter pairs.
- qaoa_problem    : QAOA-inspired feature map with cost + mixer unitaries.
- amplitude       : not supported — use QiskitExporter instead.

Requires: pip install quprep[cirq]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cirq


class CirqExporter:
    """
    Export EncodedResult objects to Cirq Circuit.

    Requires: pip install quprep[cirq]
    """

    def __init__(self):
        self._check_cirq()

    def _check_cirq(self):
        try:
            import cirq  # noqa: F401
        except ImportError:
            raise ImportError(
                "Cirq is not installed. Run: pip install quprep[cirq]"
            ) from None

    def export(self, encoded) -> cirq.Circuit:
        """Convert an EncodedResult to a Cirq Circuit.

        Parameters
        ----------
        encoded : EncodedResult
            Output from any QuPrep encoder.

        Returns
        -------
        cirq.Circuit
            Circuit using ``cirq.LineQubit`` wires.
        """
        import cirq

        encoding = encoded.metadata.get("encoding", "unknown")
        n = encoded.metadata["n_qubits"]
        params = encoded.parameters
        qubits = cirq.LineQubit.range(n)
        ops = []

        if encoding == "angle":
            rotation = encoded.metadata.get("rotation", "ry")
            gate_cls = {"ry": cirq.Ry, "rx": cirq.Rx, "rz": cirq.Rz}.get(rotation)
            if gate_cls is None:
                raise ValueError(f"Unknown rotation '{rotation}'.")
            for i, angle in enumerate(params):
                ops.append(gate_cls(rads=float(angle))(qubits[i]))

        elif encoding == "entangled_angle":
            rotation = encoded.metadata.get("rotation", "ry")
            layers = encoded.metadata.get("layers", 1)
            cnot_pairs = encoded.metadata.get("cnot_pairs", [])
            gate_cls = {"ry": cirq.Ry, "rx": cirq.Rx, "rz": cirq.Rz}.get(rotation)
            if gate_cls is None:
                raise ValueError(f"Unknown rotation '{rotation}'.")
            for _ in range(layers):
                for i, angle in enumerate(params):
                    ops.append(gate_cls(rads=float(angle))(qubits[i]))
                for ctrl, tgt in cnot_pairs:
                    ops.append(cirq.CNOT(qubits[ctrl], qubits[tgt]))

        elif encoding == "basis":
            for i, bit in enumerate(params):
                if bit == 1.0:
                    ops.append(cirq.X(qubits[i]))

        elif encoding == "amplitude":
            raise NotImplementedError(
                "Amplitude encoding requires exponential-depth state preparation "
                "which Cirq does not natively support as a simple gate. "
                "Use QiskitExporter for amplitude encoding."
            )

        elif encoding == "iqp":
            d = n
            reps = encoded.metadata.get("reps", 2)
            x = params[:d]
            pair_angles = params[d:]
            for _ in range(reps):
                ops.extend(cirq.H(qubits[i]) for i in range(d))
                ops.extend(cirq.Rz(rads=float(x[i]))(qubits[i]) for i in range(d))
                idx = 0
                for i in range(d):
                    for j in range(i + 1, d):
                        angle = float(pair_angles[idx])
                        ops.append(cirq.CNOT(qubits[i], qubits[j]))
                        ops.append(cirq.Rz(rads=angle)(qubits[j]))
                        ops.append(cirq.CNOT(qubits[i], qubits[j]))
                        idx += 1

        elif encoding == "reupload":
            layers = encoded.metadata.get("layers", 3)
            rotation = encoded.metadata.get("rotation", "ry")
            gate_cls = {"ry": cirq.Ry, "rx": cirq.Rx, "rz": cirq.Rz}.get(rotation)
            if gate_cls is None:
                raise ValueError(f"Unknown rotation '{rotation}'.")
            for _ in range(layers):
                for i, angle in enumerate(params):
                    ops.append(gate_cls(rads=float(angle))(qubits[i]))

        elif encoding == "hamiltonian":
            trotter_steps = encoded.metadata.get("trotter_steps", 4)
            for _ in range(trotter_steps):
                for i, angle in enumerate(params):
                    ops.append(cirq.Rz(rads=float(angle))(qubits[i]))

        elif encoding == "zz_feature_map":
            reps = encoded.metadata.get("reps", 2)
            single_angles = encoded.metadata["single_angles"]
            pair_angles = encoded.metadata["pair_angles"]
            pairs = encoded.metadata["pairs"]
            for _ in range(reps):
                ops.extend(cirq.H(qubits[i]) for i in range(n))
                for i, angle in enumerate(single_angles):
                    ops.append(cirq.Rz(rads=float(angle))(qubits[i]))
                for (i, j), angle in zip(pairs, pair_angles):
                    ops.append(cirq.CNOT(qubits[i], qubits[j]))
                    ops.append(cirq.Rz(rads=float(angle))(qubits[j]))
                    ops.append(cirq.CNOT(qubits[i], qubits[j]))

        elif encoding == "pauli_feature_map":
            reps = encoded.metadata.get("reps", 2)
            single_terms = encoded.metadata.get("single_terms", {})
            pair_terms = encoded.metadata.get("pair_terms", {})
            _cirq_gate = {"X": cirq.Rx, "Y": cirq.Ry, "Z": cirq.Rz}
            for _ in range(reps):
                ops.extend(cirq.H(qubits[i]) for i in range(n))
                for pauli, angles in single_terms.items():
                    gate_cls = _cirq_gate[pauli]
                    for i, angle in enumerate(angles):
                        ops.append(gate_cls(rads=float(angle))(qubits[i]))
                for pauli, entries in pair_terms.items():
                    for i, j, angle in entries:
                        if pauli == "XX":
                            ops.append(cirq.H(qubits[i]))
                            ops.append(cirq.H(qubits[j]))
                        elif pauli == "YY":
                            ops.append((cirq.S**-1)(qubits[i]))
                            ops.append((cirq.S**-1)(qubits[j]))
                        ops.append(cirq.CNOT(qubits[i], qubits[j]))
                        ops.append(cirq.Rz(rads=float(angle))(qubits[j]))
                        ops.append(cirq.CNOT(qubits[i], qubits[j]))
                        if pauli == "XX":
                            ops.append(cirq.H(qubits[i]))
                            ops.append(cirq.H(qubits[j]))
                        elif pauli == "YY":
                            ops.append(cirq.S(qubits[i]))
                            ops.append(cirq.S(qubits[j]))

        elif encoding == "random_fourier":
            rotation = encoded.metadata.get("rotation", "ry")
            gate_cls = {"ry": cirq.Ry, "rx": cirq.Rx, "rz": cirq.Rz}.get(rotation)
            if gate_cls is None:
                raise ValueError(f"Unknown rotation '{rotation}'.")
            for i, angle in enumerate(params):
                ops.append(gate_cls(rads=float(angle))(qubits[i]))

        elif encoding == "tensor_product":
            ry_angles = encoded.metadata["ry_angles"]
            rz_angles = encoded.metadata["rz_angles"]
            for k in range(n):
                ops.append(cirq.Ry(rads=float(ry_angles[k]))(qubits[k]))
                ops.append(cirq.Rz(rads=float(rz_angles[k]))(qubits[k]))

        elif encoding == "qaoa_problem":
            p = encoded.metadata.get("p", 1)
            beta = encoded.metadata.get("beta", 0.39269908169872414)
            local_angles = encoded.metadata["local_angles"]
            coupling_angles = encoded.metadata["coupling_angles"]
            pairs = encoded.metadata.get("pairs", [])
            ops.extend(cirq.H(qubits[i]) for i in range(n))
            for _ in range(p):
                for i in range(n):
                    ops.append(cirq.Rz(rads=2.0 * float(local_angles[i]))(qubits[i]))
                for k, (i, j) in enumerate(pairs):
                    ops.append(cirq.CNOT(qubits[i], qubits[j]))
                    ops.append(cirq.Rz(rads=2.0 * float(coupling_angles[k]))(qubits[j]))
                    ops.append(cirq.CNOT(qubits[i], qubits[j]))
                for i in range(n):
                    ops.append(cirq.Rx(rads=2.0 * float(beta))(qubits[i]))

        else:
            raise ValueError(
                f"Unknown encoding '{encoding}'. "
                "Supported: angle, entangled_angle, basis, iqp, reupload, hamiltonian, "
                "zz_feature_map, pauli_feature_map, random_fourier, tensor_product, qaoa_problem."
            )

        return cirq.Circuit(ops)

    def export_batch(self, encoded_list: list) -> list:
        """
        Export a list of EncodedResults to Cirq Circuits.

        Parameters
        ----------
        encoded_list : list of EncodedResult

        Returns
        -------
        list of cirq.Circuit
            One circuit per sample.
        """
        return [self.export(e) for e in encoded_list]
