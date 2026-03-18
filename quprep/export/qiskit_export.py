"""Export encoded data as Qiskit QuantumCircuit objects."""

from __future__ import annotations


class QiskitExporter:
    """
    Export EncodedResult objects to Qiskit QuantumCircuit.

    Requires: pip install quprep[qiskit]

    Parameters
    ----------
    backend : str, optional
        IBM backend name (e.g. 'ibm_brisbane'). Used to add transpilation hints.
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

    def export(self, encoded) -> "qiskit.QuantumCircuit":
        """Convert an EncodedResult to a Qiskit QuantumCircuit."""
        raise NotImplementedError("QiskitExporter.export() — coming in v0.1.0")

    def export_batch(self, encoded_list: list) -> list:
        """Export a list of EncodedResults."""
        return [self.export(e) for e in encoded_list]
