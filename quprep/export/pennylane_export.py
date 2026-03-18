"""Export encoded data as PennyLane QNode-compatible circuits."""

from __future__ import annotations


class PennyLaneExporter:
    """
    Export EncodedResult objects to PennyLane.

    Requires: pip install quprep[pennylane]

    Parameters
    ----------
    interface : str
        Autodiff interface: 'torch', 'jax', 'tf', or 'auto'. Default 'auto'.
    device : str
        PennyLane device string. Default 'default.qubit'.
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

    def export(self, encoded):
        """Return a PennyLane QNode template for an EncodedResult."""
        raise NotImplementedError("PennyLaneExporter.export() — coming in v0.2.0")
