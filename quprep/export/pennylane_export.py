"""Export encoded data as PennyLane QNode-compatible circuits.

Supported encodings
-------------------
- angle           : RY/RX/RZ gate per qubit.
- entangled_angle : rotation layer + CNOT entangling layer, repeated layers times.
- basis           : PauliX gates on qubits where the bit is 1.
- amplitude       : AmplitudeEmbedding (full state vector initialization).
- iqp             : Hadamards + RZ(x_i) + IsingZZ(x_i·x_j) interactions, repeated reps times.
- reupload        : rotation gate repeated ``layers`` times per qubit.
- hamiltonian     : RZ(2·x_i·T/S) per qubit, repeated trotter_steps times.
- zz_feature_map  : H + Rz single-qubit + CNOT-Rz-CNOT pairwise, repeated reps times.
- pauli_feature_map: Generalised Pauli feature map (X/Y/Z, XX/YY/ZZ strings).
- random_fourier  : angle-encoded Fourier features (treated as angle encoding).
- tensor_product  : Ry + Rz per qubit from alternating parameter pairs.
- qaoa_problem    : QAOA-inspired feature map with cost + mixer unitaries.

Requires: pip install quprep[pennylane]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pennylane as qml


class PennyLaneExporter:
    """
    Export EncodedResult objects to PennyLane QNode circuits.

    Returns a callable ``qml.QNode`` from :meth:`export`. Calling the returned
    QNode executes the circuit on the configured device and returns the
    quantum state.

    Requires: pip install quprep[pennylane]

    Parameters
    ----------
    interface : str
        Autodiff interface: ``'torch'``, ``'jax'``, ``'tf'``, or ``'auto'``.
        Default ``'auto'``.
    device : str
        PennyLane device string. Default ``'default.qubit'``.
    """

    def __init__(self, interface: str = "auto", device: str = "default.qubit"):
        self.interface = interface
        self.device = device
        self._check_pennylane()

    def _check_pennylane(self):
        try:
            import pennylane  # noqa: F401
        except ImportError:
            raise ImportError(
                "PennyLane is not installed. Run: pip install quprep[pennylane]"
            ) from None

    def export(self, encoded) -> qml.QNode:
        """Return a PennyLane QNode for an EncodedResult.

        The returned QNode is immediately callable — invoke it with no
        arguments to execute the circuit and obtain the quantum state.

        Parameters
        ----------
        encoded : EncodedResult
            Output from any QuPrep encoder. Supports all 12 encodings:
            ``angle``, ``entangled_angle``, ``basis``, ``amplitude``,
            ``iqp``, ``reupload``, ``hamiltonian``, ``zz_feature_map``,
            ``pauli_feature_map``, ``random_fourier``, ``tensor_product``,
            ``qaoa_problem``.

        Returns
        -------
        qml.QNode
            Callable quantum circuit. Invoke with no arguments to execute
            and obtain the quantum state vector.
        """
        import pennylane as qml

        encoding = encoded.metadata.get("encoding", "unknown")
        n = encoded.metadata["n_qubits"]
        params = encoded.parameters.copy()
        dev = qml.device(self.device, wires=n)

        if encoding == "angle":
            rotation = encoded.metadata.get("rotation", "ry")
            gate = {"ry": qml.RY, "rx": qml.RX, "rz": qml.RZ}.get(rotation)
            if gate is None:
                raise ValueError(f"Unknown rotation '{rotation}'.")

            @qml.qnode(dev, interface=self.interface)
            def circuit():
                for i, angle in enumerate(params):
                    gate(float(angle), wires=i)
                return qml.state()

        elif encoding == "entangled_angle":
            rotation = encoded.metadata.get("rotation", "ry")
            layers = encoded.metadata.get("layers", 1)
            cnot_pairs = encoded.metadata.get("cnot_pairs", [])
            gate = {"ry": qml.RY, "rx": qml.RX, "rz": qml.RZ}.get(rotation)
            if gate is None:
                raise ValueError(f"Unknown rotation '{rotation}'.")

            @qml.qnode(dev, interface=self.interface)
            def circuit():
                for _ in range(layers):
                    for i, angle in enumerate(params):
                        gate(float(angle), wires=i)
                    for ctrl, tgt in cnot_pairs:
                        qml.CNOT(wires=[ctrl, tgt])
                return qml.state()

        elif encoding == "basis":
            @qml.qnode(dev, interface=self.interface)
            def circuit():
                for i, bit in enumerate(params):
                    if bit == 1.0:
                        qml.PauliX(wires=i)
                return qml.state()

        elif encoding == "amplitude":
            @qml.qnode(dev, interface=self.interface)
            def circuit():
                qml.AmplitudeEmbedding(params, wires=range(n), normalize=False)
                return qml.state()

        elif encoding == "iqp":
            d = n
            reps = encoded.metadata.get("reps", 2)
            x = params[:d]
            pair_angles = params[d:]

            @qml.qnode(dev, interface=self.interface)
            def circuit():
                for _ in range(reps):
                    for i in range(d):
                        qml.Hadamard(wires=i)
                    for i in range(d):
                        qml.RZ(float(x[i]), wires=i)
                    idx = 0
                    for i in range(d):
                        for j in range(i + 1, d):
                            qml.IsingZZ(float(pair_angles[idx]), wires=[i, j])
                            idx += 1
                return qml.state()

        elif encoding == "reupload":
            layers = encoded.metadata.get("layers", 3)
            rotation = encoded.metadata.get("rotation", "ry")
            gate = {"ry": qml.RY, "rx": qml.RX, "rz": qml.RZ}.get(rotation)
            if gate is None:
                raise ValueError(f"Unknown rotation '{rotation}'.")

            @qml.qnode(dev, interface=self.interface)
            def circuit():
                for _ in range(layers):
                    for i, angle in enumerate(params):
                        gate(float(angle), wires=i)
                return qml.state()

        elif encoding == "hamiltonian":
            trotter_steps = encoded.metadata.get("trotter_steps", 4)

            @qml.qnode(dev, interface=self.interface)
            def circuit():
                for _ in range(trotter_steps):
                    for i, angle in enumerate(params):
                        qml.RZ(float(angle), wires=i)
                return qml.state()

        elif encoding == "zz_feature_map":
            reps = encoded.metadata.get("reps", 2)
            single_angles = encoded.metadata["single_angles"]
            pair_angles = encoded.metadata["pair_angles"]
            pairs = encoded.metadata["pairs"]

            @qml.qnode(dev, interface=self.interface)
            def circuit():
                for _ in range(reps):
                    for i in range(n):
                        qml.Hadamard(wires=i)
                    for i, angle in enumerate(single_angles):
                        qml.RZ(float(angle), wires=i)
                    for (i, j), angle in zip(pairs, pair_angles):
                        qml.CNOT(wires=[i, j])
                        qml.RZ(float(angle), wires=j)
                        qml.CNOT(wires=[i, j])
                return qml.state()

        elif encoding == "pauli_feature_map":
            reps = encoded.metadata.get("reps", 2)
            single_terms = encoded.metadata.get("single_terms", {})
            pair_terms = encoded.metadata.get("pair_terms", {})

            @qml.qnode(dev, interface=self.interface)
            def circuit():
                _pl_gate = {"X": qml.RX, "Y": qml.RY, "Z": qml.RZ}
                for _ in range(reps):
                    for i in range(n):
                        qml.Hadamard(wires=i)
                    for pauli, angles in single_terms.items():
                        gate = _pl_gate[pauli]
                        for i, angle in enumerate(angles):
                            gate(float(angle), wires=i)
                    for pauli, entries in pair_terms.items():
                        for i, j, angle in entries:
                            if pauli == "XX":
                                qml.Hadamard(wires=i)
                                qml.Hadamard(wires=j)
                            elif pauli == "YY":
                                qml.adjoint(qml.S)(wires=i)
                                qml.adjoint(qml.S)(wires=j)
                            qml.CNOT(wires=[i, j])
                            qml.RZ(float(angle), wires=j)
                            qml.CNOT(wires=[i, j])
                            if pauli == "XX":
                                qml.Hadamard(wires=i)
                                qml.Hadamard(wires=j)
                            elif pauli == "YY":
                                qml.S(wires=i)
                                qml.S(wires=j)
                return qml.state()

        elif encoding == "random_fourier":
            rotation = encoded.metadata.get("rotation", "ry")
            gate = {"ry": qml.RY, "rx": qml.RX, "rz": qml.RZ}.get(rotation)
            if gate is None:
                raise ValueError(f"Unknown rotation '{rotation}'.")

            @qml.qnode(dev, interface=self.interface)
            def circuit():
                for i, angle in enumerate(params):
                    gate(float(angle), wires=i)
                return qml.state()

        elif encoding == "tensor_product":
            ry_angles = encoded.metadata["ry_angles"]
            rz_angles = encoded.metadata["rz_angles"]

            @qml.qnode(dev, interface=self.interface)
            def circuit():
                for k in range(n):
                    qml.RY(float(ry_angles[k]), wires=k)
                    qml.RZ(float(rz_angles[k]), wires=k)
                return qml.state()

        elif encoding == "qaoa_problem":
            p = encoded.metadata.get("p", 1)
            beta = encoded.metadata.get("beta", 0.39269908169872414)
            local_angles = encoded.metadata["local_angles"]
            coupling_angles = encoded.metadata["coupling_angles"]
            pairs = encoded.metadata.get("pairs", [])

            @qml.qnode(dev, interface=self.interface)
            def circuit():
                for i in range(n):
                    qml.Hadamard(wires=i)
                for _ in range(p):
                    for i in range(n):
                        qml.RZ(2.0 * float(local_angles[i]), wires=i)
                    for k, (i, j) in enumerate(pairs):
                        qml.CNOT(wires=[i, j])
                        qml.RZ(2.0 * float(coupling_angles[k]), wires=j)
                        qml.CNOT(wires=[i, j])
                    for i in range(n):
                        qml.RX(2.0 * float(beta), wires=i)
                return qml.state()

        else:
            raise ValueError(
                f"Unknown encoding '{encoding}'. "
                "Supported: angle, entangled_angle, basis, amplitude, iqp, reupload, "
                "hamiltonian, zz_feature_map, pauli_feature_map, random_fourier, "
                "tensor_product, qaoa_problem."
            )

        return circuit

    def export_batch(self, encoded_list: list) -> list:
        """
        Export a list of EncodedResults to PennyLane QNodes.

        Parameters
        ----------
        encoded_list : list of EncodedResult

        Returns
        -------
        list of qml.QNode
            One callable QNode per sample.
        """
        return [self.export(e) for e in encoded_list]
