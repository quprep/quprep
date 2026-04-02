"""Export encoded data as Amazon Braket Circuit objects.

Supported encodings
-------------------
- angle           : Ry/Rx/Rz gate per qubit.
- entangled_angle : rotation layer + CNOT entangling layer, repeated layers times.
- basis           : X gates on qubits where the bit is 1.
- iqp             : H + Rz(x_i) + ZZ(x_i·x_j) interactions, repeated reps times.
- zz_feature_map  : H + Rz single-qubit + CNOT-Rz-CNOT pairwise, repeated reps times.
- pauli_feature_map: Generalised Pauli feature map (X/Y/Z, XX/YY/ZZ strings).
- random_fourier  : angle-encoded Fourier features (treated as angle encoding).
- reupload        : rotation gate repeated `layers` times per qubit.
- hamiltonian     : Rz(2·x_i·T/S) per qubit, repeated trotter_steps times.
- tensor_product  : Ry + Rz per qubit from alternating parameter pairs.
- qaoa_problem    : QAOA-inspired feature map with cost + mixer unitaries.
- amplitude       : not supported — use QiskitExporter instead.

Requires: pip install quprep[braket]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from braket.circuits import Circuit


class BraketExporter:
    """
    Export EncodedResult objects to Amazon Braket Circuit objects.

    Requires: pip install quprep[braket]
    """

    def __init__(self):
        self._check_braket()

    def _check_braket(self):
        try:
            import braket.circuits  # noqa: F401
        except ImportError:
            raise ImportError(
                "amazon-braket-sdk is not installed. Run: pip install quprep[braket]"
            ) from None

    def export(self, encoded) -> Circuit:
        """
        Convert an EncodedResult to a Braket Circuit.

        Parameters
        ----------
        encoded : EncodedResult
            Output from any QuPrep encoder (except amplitude).

        Returns
        -------
        braket.circuits.Circuit
        """
        from braket.circuits import Circuit, Instruction, gates

        encoding = encoded.metadata.get("encoding", "unknown")
        params = encoded.parameters

        if encoding == "angle":
            n = encoded.metadata["n_qubits"]
            circ = Circuit()
            rotation = encoded.metadata.get("rotation", "ry")
            gate_fn = {"ry": gates.Ry, "rx": gates.Rx, "rz": gates.Rz}.get(rotation)
            if gate_fn is None:
                raise ValueError(f"Unknown rotation '{rotation}'.")
            for i, angle in enumerate(params):
                circ.add_instruction(Instruction(gate_fn(float(angle)), i))
            return circ

        if encoding == "entangled_angle":
            n = encoded.metadata["n_qubits"]
            rotation = encoded.metadata.get("rotation", "ry")
            layers = encoded.metadata.get("layers", 1)
            cnot_pairs = encoded.metadata.get("cnot_pairs", [])
            circ = Circuit()
            gate_fn = {"ry": gates.Ry, "rx": gates.Rx, "rz": gates.Rz}.get(rotation)
            if gate_fn is None:
                raise ValueError(f"Unknown rotation '{rotation}'.")
            for _ in range(layers):
                for i, angle in enumerate(params):
                    circ.add_instruction(Instruction(gate_fn(float(angle)), i))
                for ctrl, tgt in cnot_pairs:
                    circ.cnot(ctrl, tgt)
            return circ

        if encoding == "basis":
            n = len(params)
            circ = Circuit()
            for i, bit in enumerate(params):
                if bit == 1.0:
                    circ.x(i)
            return circ

        if encoding == "iqp":
            d = encoded.metadata["n_qubits"]
            reps = encoded.metadata.get("reps", 2)
            x = params[:d]
            pair_angles = params[d:]
            circ = Circuit()
            for _ in range(reps):
                for i in range(d):
                    circ.h(i)
                for i in range(d):
                    circ.add_instruction(Instruction(gates.Rz(float(x[i])), i))
                idx = 0
                for i in range(d):
                    for j in range(i + 1, d):
                        angle = float(pair_angles[idx])
                        circ.cnot(i, j)
                        circ.add_instruction(Instruction(gates.Rz(angle), j))
                        circ.cnot(i, j)
                        idx += 1
            return circ

        if encoding == "zz_feature_map":
            d = encoded.metadata["n_qubits"]
            reps = encoded.metadata.get("reps", 2)
            single_angles = encoded.metadata["single_angles"]
            pair_angles = encoded.metadata["pair_angles"]
            pairs = encoded.metadata["pairs"]
            circ = Circuit()
            for _ in range(reps):
                for i in range(d):
                    circ.h(i)
                for i, angle in enumerate(single_angles):
                    circ.add_instruction(Instruction(gates.Rz(float(angle)), i))
                for (i, j), angle in zip(pairs, pair_angles):
                    circ.cnot(i, j)
                    circ.add_instruction(Instruction(gates.Rz(float(angle)), j))
                    circ.cnot(i, j)
            return circ

        if encoding == "tensor_product":
            n = encoded.metadata["n_qubits"]
            ry_angles = encoded.metadata["ry_angles"]
            rz_angles = encoded.metadata["rz_angles"]
            circ = Circuit()
            for k in range(n):
                circ.add_instruction(Instruction(gates.Ry(float(ry_angles[k])), k))
                circ.add_instruction(Instruction(gates.Rz(float(rz_angles[k])), k))
            return circ

        if encoding == "reupload":
            n = len(params)
            layers = encoded.metadata.get("layers", 3)
            rotation = encoded.metadata.get("rotation", "ry")
            circ = Circuit()
            gate_fn = {"ry": gates.Ry, "rx": gates.Rx, "rz": gates.Rz}.get(rotation)
            if gate_fn is None:
                raise ValueError(f"Unknown rotation '{rotation}'.")
            for _ in range(layers):
                for i, angle in enumerate(params):
                    circ.add_instruction(Instruction(gate_fn(float(angle)), i))
            return circ

        if encoding == "hamiltonian":
            n = len(params)
            trotter_steps = encoded.metadata.get("trotter_steps", 4)
            circ = Circuit()
            for _ in range(trotter_steps):
                for i, angle in enumerate(params):
                    circ.add_instruction(Instruction(gates.Rz(float(angle)), i))
            return circ

        if encoding == "qaoa_problem":
            d = encoded.metadata["n_qubits"]
            p = encoded.metadata.get("p", 1)
            beta = encoded.metadata.get("beta", 0.39269908169872414)
            local_angles = encoded.metadata["local_angles"]
            coupling_angles = encoded.metadata["coupling_angles"]
            pairs = encoded.metadata.get("pairs", [])
            circ = Circuit()
            for i in range(d):
                circ.h(i)
            for _ in range(p):
                for i in range(d):
                    circ.add_instruction(Instruction(gates.Rz(2.0 * float(local_angles[i])), i))
                for k, (i, j) in enumerate(pairs):
                    angle = 2.0 * float(coupling_angles[k])
                    circ.cnot(i, j)
                    circ.add_instruction(Instruction(gates.Rz(angle), j))
                    circ.cnot(i, j)
                for i in range(d):
                    circ.add_instruction(Instruction(gates.Rx(2.0 * float(beta)), i))
            return circ

        if encoding == "pauli_feature_map":
            d = encoded.metadata["n_qubits"]
            reps = encoded.metadata.get("reps", 2)
            single_terms = encoded.metadata.get("single_terms", {})
            pair_terms = encoded.metadata.get("pair_terms", {})
            circ = Circuit()
            _gate = {"X": gates.Rx, "Y": gates.Ry, "Z": gates.Rz}
            for _ in range(reps):
                for i in range(d):
                    circ.h(i)
                for pauli, angles in single_terms.items():
                    gate_fn = _gate[pauli]
                    for i, angle in enumerate(angles):
                        circ.add_instruction(Instruction(gate_fn(float(angle)), i))
                for pauli, entries in pair_terms.items():
                    for i, j, angle in entries:
                        if pauli == "XX":
                            circ.h(i)
                            circ.h(j)
                        elif pauli == "YY":
                            circ.add_instruction(Instruction(gates.Si(), i))
                            circ.add_instruction(Instruction(gates.Si(), j))
                        circ.cnot(i, j)
                        circ.add_instruction(Instruction(gates.Rz(float(angle)), j))
                        circ.cnot(i, j)
                        if pauli == "XX":
                            circ.h(i)
                            circ.h(j)
                        elif pauli == "YY":
                            circ.add_instruction(Instruction(gates.S(), i))
                            circ.add_instruction(Instruction(gates.S(), j))
            return circ

        if encoding == "random_fourier":
            n = encoded.metadata["n_qubits"]
            circ = Circuit()
            rotation = encoded.metadata.get("rotation", "ry")
            gate_fn = {"ry": gates.Ry, "rx": gates.Rx, "rz": gates.Rz}.get(rotation)
            if gate_fn is None:
                raise ValueError(f"Unknown rotation '{rotation}'.")
            for i, angle in enumerate(params):
                circ.add_instruction(Instruction(gate_fn(float(angle)), i))
            return circ

        if encoding == "amplitude":
            raise NotImplementedError(
                "Amplitude encoding requires exponential-depth state preparation. "
                "Use QiskitExporter for amplitude encoding."
            )

        raise ValueError(
            f"Unknown encoding '{encoding}'. "
            "Supported: angle, entangled_angle, basis, iqp, zz_feature_map, "
            "pauli_feature_map, random_fourier, tensor_product, qaoa_problem, "
            "reupload, hamiltonian."
        )

    def export_batch(self, encoded_list: list) -> list:
        """
        Export a list of EncodedResults to Braket Circuits.

        Parameters
        ----------
        encoded_list : list of EncodedResult

        Returns
        -------
        list of braket.circuits.Circuit
        """
        return [self.export(e) for e in encoded_list]
