"""Export encoded data as Cirq Circuit objects.

Supported encodings
-------------------
- angle       : Ry/Rx/Rz gate per qubit.
- basis       : X gates on qubits where the bit is 1.
- iqp         : H + Rz(x_i) + CX+Rz(x_i·x_j)+CX interactions, repeated reps times.
- reupload    : rotation gate repeated ``layers`` times per qubit.
- hamiltonian : Rz(2·x_i·T/S) per qubit, repeated trotter_steps times.
- amplitude   : not supported — use QiskitExporter instead.

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

        else:
            raise ValueError(
                f"Unknown encoding '{encoding}'. "
                "Supported: angle, basis, iqp, reupload, hamiltonian."
            )

        return cirq.Circuit(ops)

    def export_batch(self, encoded_list: list) -> list:
        """Export a list of EncodedResults to Cirq Circuits."""
        return [self.export(e) for e in encoded_list]
