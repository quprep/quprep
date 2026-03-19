"""Export encoded data as Qiskit QuantumCircuit objects.

Supported encodings
-------------------
- angle     : ry/rx/rz gate per qubit.
- basis     : X gates on qubits where the bit is 1.
- amplitude : StatePreparation gate (full state vector initialization).

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
        """Convert an EncodedResult to a Qiskit QuantumCircuit."""
        from qiskit import QuantumCircuit

        encoding = encoded.metadata.get("encoding", "unknown")
        n_qubits = encoded.metadata["n_qubits"]
        params = encoded.parameters

        qc = QuantumCircuit(n_qubits)

        if encoding == "angle":
            rotation = encoded.metadata.get("rotation", "ry")
            gate_fn = getattr(qc, rotation)
            for i, angle in enumerate(params):
                gate_fn(float(angle), i)

        elif encoding == "basis":
            for i, bit in enumerate(params):
                if bit == 1.0:
                    qc.x(i)

        elif encoding == "amplitude":
            from qiskit.circuit.library import StatePreparation
            qc.append(StatePreparation(params.tolist()), range(n_qubits))

        else:
            raise ValueError(
                f"Unknown encoding '{encoding}'. "
                "QiskitExporter supports 'angle', 'basis', and 'amplitude'."
            )

        return qc

    def export_batch(self, encoded_list: list) -> list:
        """Export a list of EncodedResults."""
        return [self.export(e) for e in encoded_list]
