"""Export encoded data as Cirq Circuit objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cirq


class CirqExporter:
    """
    Export EncodedResult objects to Cirq.

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

    def export(self, encoded) -> cirq.Circuit:  # type: ignore[return]
        raise NotImplementedError("CirqExporter.export() — coming in v0.2.0")
