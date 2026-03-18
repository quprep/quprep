"""Export encoded data as TKET (pytket) Circuit objects."""

from __future__ import annotations


class TKETExporter:
    """
    Export EncodedResult objects to TKET/pytket.

    Requires: pip install quprep[tket]
    """

    def __init__(self):
        self._check_pytket()

    def _check_pytket(self):
        try:
            import pytket  # noqa: F401
        except ImportError:
            raise ImportError(
                "pytket is not installed. Run: pip install quprep[tket]"
            ) from None

    def export(self, encoded) -> "pytket.Circuit":
        raise NotImplementedError("TKETExporter.export() — coming in v0.2.0")
