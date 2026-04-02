"""Export encoded data as Qiskit QuantumCircuit objects.

Supported encodings
-------------------
- angle           : ry/rx/rz gate per qubit.
- entangled_angle : rotation layer + CNOT entangling layer, repeated layers times.
- basis           : X gates on qubits where the bit is 1.
- iqp             : H + Rz(x_i) + ZZ(x_i·x_j) interactions, repeated reps times.
- reupload        : rotation gate repeated `layers` times per qubit.
- hamiltonian     : Rz(angle) per qubit, repeated trotter_steps times.
- zz_feature_map  : H + Rz single-qubit + CNOT-Rz-CNOT pairwise, repeated reps times.
- pauli_feature_map: Generalised Pauli feature map (X/Y/Z, XX/YY/ZZ strings).
- random_fourier  : angle-encoded Fourier features (treated as angle encoding).
- tensor_product  : Ry + Rz per qubit from alternating parameter pairs.
- qaoa_problem    : QAOA-inspired feature map with cost + mixer unitaries.
- amplitude       : StatePreparation gate (full state vector initialization).

Requires: pip install quprep[qiskit]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import qiskit


class QiskitExporter:
    """
    Export EncodedResult objects to Qiskit QuantumCircuit.

    Requires: pip install quprep[qiskit]

    Parameters
    ----------
    backend : str, optional
        IBM backend name (e.g. 'ibm_brisbane'). Reserved for transpilation hints
        in a future release — currently stored but not applied.
    """

    def __init__(self, backend: str | None = None):
        self.backend = backend
        self._check_qiskit()

    def _check_qiskit(self):
        try:
            import qiskit  # noqa: F401
        except ImportError:
            raise ImportError(
                "Qiskit is not installed. Run: pip install quprep[qiskit]"
            ) from None

    def export(self, encoded) -> qiskit.QuantumCircuit:
        """
        Convert an EncodedResult to a Qiskit QuantumCircuit.

        Parameters
        ----------
        encoded : EncodedResult
            Output from any QuPrep encoder.

        Returns
        -------
        qiskit.QuantumCircuit
            Ready-to-transpile Qiskit circuit.
        """
        from qiskit import QuantumCircuit

        encoding = encoded.metadata.get("encoding", "unknown")
        n_qubits = encoded.metadata["n_qubits"]
        params = encoded.parameters

        qc = QuantumCircuit(n_qubits)

        if encoding == "angle" or encoding == "random_fourier":
            rotation = encoded.metadata.get("rotation", "ry")
            gate_fn = getattr(qc, rotation)
            for i, angle in enumerate(params):
                gate_fn(float(angle), i)

        elif encoding == "entangled_angle":
            rotation = encoded.metadata.get("rotation", "ry")
            layers = encoded.metadata.get("layers", 1)
            cnot_pairs = encoded.metadata.get("cnot_pairs", [])
            gate_fn = getattr(qc, rotation)
            for _ in range(layers):
                for i, angle in enumerate(params):
                    gate_fn(float(angle), i)
                for ctrl, tgt in cnot_pairs:
                    qc.cx(ctrl, tgt)

        elif encoding == "basis":
            for i, bit in enumerate(params):
                if bit == 1.0:
                    qc.x(i)

        elif encoding == "iqp":
            d = n_qubits
            reps = encoded.metadata.get("reps", 2)
            x = params[:d]
            pair_angles = params[d:]
            for _ in range(reps):
                for i in range(d):
                    qc.h(i)
                for i in range(d):
                    qc.rz(float(x[i]), i)
                idx = 0
                for i in range(d):
                    for j in range(i + 1, d):
                        angle = float(pair_angles[idx])
                        qc.cx(i, j)
                        qc.rz(angle, j)
                        qc.cx(i, j)
                        idx += 1

        elif encoding == "reupload":
            layers = encoded.metadata.get("layers", 3)
            rotation = encoded.metadata.get("rotation", "ry")
            gate_fn = getattr(qc, rotation)
            for _ in range(layers):
                for i, angle in enumerate(params):
                    gate_fn(float(angle), i)

        elif encoding == "hamiltonian":
            trotter_steps = encoded.metadata.get("trotter_steps", 4)
            for _ in range(trotter_steps):
                for i, angle in enumerate(params):
                    qc.rz(float(angle), i)

        elif encoding == "zz_feature_map":
            reps = encoded.metadata.get("reps", 2)
            single_angles = encoded.metadata["single_angles"]
            pair_angles = encoded.metadata["pair_angles"]
            pairs = encoded.metadata["pairs"]
            for _ in range(reps):
                for i in range(n_qubits):
                    qc.h(i)
                for i, angle in enumerate(single_angles):
                    qc.rz(float(angle), i)
                for (i, j), angle in zip(pairs, pair_angles):
                    qc.cx(i, j)
                    qc.rz(float(angle), j)
                    qc.cx(i, j)

        elif encoding == "pauli_feature_map":
            reps = encoded.metadata.get("reps", 2)
            single_terms = encoded.metadata.get("single_terms", {})
            pair_terms = encoded.metadata.get("pair_terms", {})
            _gate = {"X": "rx", "Y": "ry", "Z": "rz"}
            _conj_pre = {"XX": "h", "YY": "sdg", "ZZ": None}
            _conj_post = {"XX": "h", "YY": "s", "ZZ": None}
            for _ in range(reps):
                for i in range(n_qubits):
                    qc.h(i)
                for pauli, angles in single_terms.items():
                    gate_fn = getattr(qc, _gate[pauli])
                    for i, angle in enumerate(angles):
                        gate_fn(float(angle), i)
                for pauli, entries in pair_terms.items():
                    pre = _conj_pre.get(pauli)
                    post = _conj_post.get(pauli)
                    for i, j, angle in entries:
                        if pre:
                            getattr(qc, pre)(i)
                            getattr(qc, pre)(j)
                        qc.cx(i, j)
                        qc.rz(float(angle), j)
                        qc.cx(i, j)
                        if post:
                            getattr(qc, post)(i)
                            getattr(qc, post)(j)

        elif encoding == "tensor_product":
            ry_angles = encoded.metadata["ry_angles"]
            rz_angles = encoded.metadata["rz_angles"]
            for k in range(n_qubits):
                qc.ry(float(ry_angles[k]), k)
                qc.rz(float(rz_angles[k]), k)

        elif encoding == "qaoa_problem":
            p = encoded.metadata.get("p", 1)
            beta = encoded.metadata.get("beta", 0.39269908169872414)
            local_angles = encoded.metadata["local_angles"]
            coupling_angles = encoded.metadata["coupling_angles"]
            pairs = encoded.metadata.get("pairs", [])
            for i in range(n_qubits):
                qc.h(i)
            for _ in range(p):
                for i in range(n_qubits):
                    qc.rz(2.0 * float(local_angles[i]), i)
                for k, (i, j) in enumerate(pairs):
                    angle = 2.0 * float(coupling_angles[k])
                    qc.cx(i, j)
                    qc.rz(angle, j)
                    qc.cx(i, j)
                for i in range(n_qubits):
                    qc.rx(2.0 * float(beta), i)

        elif encoding == "amplitude":
            from qiskit.circuit.library import StatePreparation
            qc.append(StatePreparation(params.tolist()), range(n_qubits))

        else:
            raise ValueError(
                f"Unknown encoding '{encoding}'. "
                "Supported: angle, entangled_angle, basis, iqp, reupload, hamiltonian, "
                "zz_feature_map, pauli_feature_map, random_fourier, tensor_product, "
                "qaoa_problem, amplitude."
            )

        return qc

    def export_batch(self, encoded_list: list) -> list:
        """
        Export a list of EncodedResults to Qiskit QuantumCircuits.

        Parameters
        ----------
        encoded_list : list of EncodedResult

        Returns
        -------
        list of qiskit.QuantumCircuit
            One circuit per sample.
        """
        return [self.export(e) for e in encoded_list]
