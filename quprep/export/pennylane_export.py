"""Export encoded data as PennyLane QNode-compatible circuits.

Supported encodings
-------------------
- angle       : RY/RX/RZ gate per qubit.
- basis       : PauliX gates on qubits where the bit is 1.
- amplitude   : AmplitudeEmbedding (full state vector initialization).
- iqp         : Hadamards + RZ(x_i) + IsingZZ(x_i·x_j) interactions, repeated reps times.
- reupload    : rotation gate repeated ``layers`` times per qubit.
- hamiltonian : RZ(2·x_i·T/S) per qubit, repeated trotter_steps times.

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
            Output from any QuPrep encoder.

        Returns
        -------
        qml.QNode
            Callable quantum circuit.
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

        else:
            raise ValueError(
                f"Unknown encoding '{encoding}'. "
                "Supported: angle, basis, amplitude, iqp, reupload, hamiltonian."
            )

        return circuit

    def export_batch(self, encoded_list: list) -> list:
        """Export a list of EncodedResults to PennyLane QNodes."""
        return [self.export(e) for e in encoded_list]
