"""Data-driven circuit quality metrics for QuPrep encodings.

All metrics operate on the actual output states produced by encoding data
through a circuit.  No quantum hardware or external framework is required —
a lightweight numpy statevector simulator handles all computation.

Simulation is limited to circuits with ``n_qubits ≤ metrics.MAX_QUBITS``
(default 12).  For larger circuits the functions return ``None``.

Functions
---------
expressibility(encoder, dataset)
    KL divergence of the output-state fidelity distribution from the Haar
    measure.  **Lower = more expressive.**

entanglement_capability(encoder, dataset)
    Average Meyer-Wallach entanglement measure over sampled data points.
    **Higher = more entangled.** 0 for product-state encodings.

kernel_alignment(encoder, dataset)
    Normalised Frobenius alignment of the quantum kernel with class labels.
    **Higher = better class separation.** Requires labelled data.

score_encoding(encoder, dataset)
    Compute all three metrics and return an ``EncoderMetrics`` dataclass.
"""

from __future__ import annotations

from quprep.metrics._simulate import MAX_QUBITS
from quprep.metrics.expressibility import entanglement_capability, expressibility
from quprep.metrics.kernel import EncoderMetrics, kernel_alignment, score_encoding

__all__ = [
    "MAX_QUBITS",
    "expressibility",
    "entanglement_capability",
    "kernel_alignment",
    "EncoderMetrics",
    "score_encoding",
]
