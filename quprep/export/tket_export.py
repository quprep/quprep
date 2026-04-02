"""Export encoded data as TKET (pytket) Circuit objects.

Supported encodings
-------------------
- angle           : Ry/Rx/Rz gate per qubit (angles in radians, converted to half-turns).
- entangled_angle : rotation layer + CX entangling layer, repeated layers times.
- basis           : X gates on qubits where the bit is 1.
- iqp             : H + Rz(x_i) + CX+Rz(x_i·x_j)+CX interactions, repeated reps times.
- reupload        : rotation gate repeated ``layers`` times per qubit.
- hamiltonian     : Rz(2·x_i·T/S) per qubit, repeated trotter_steps times.
- zz_feature_map  : H + Rz single-qubit + CX-Rz-CX pairwise, repeated reps times.
- pauli_feature_map: Generalised Pauli feature map (X/Y/Z, XX/YY/ZZ strings).
- random_fourier  : angle-encoded Fourier features (treated as angle encoding).
- tensor_product  : Ry + Rz per qubit from alternating parameter pairs.
- qaoa_problem    : QAOA-inspired feature map with cost + mixer unitaries.
- amplitude       : not supported — use QiskitExporter instead.

Requires: pip install quprep[tket]
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytket


class TKETExporter:
    r"""
    Export EncodedResult objects to TKET/pytket Circuit.

    pytket rotation gates use half-turns ($\text{angle} / \pi$). This exporter converts
    all radian angles from QuPrep encoders to the pytket convention automatically.

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

    def export(self, encoded) -> pytket.Circuit:
        """Convert an EncodedResult to a pytket Circuit.

        Parameters
        ----------
        encoded : EncodedResult
            Output from any QuPrep encoder.

        Returns
        -------
        pytket.Circuit
            Circuit with angles converted to pytket half-turns.
        """
        from pytket import Circuit

        encoding = encoded.metadata.get("encoding", "unknown")
        n = encoded.metadata["n_qubits"]
        params = encoded.parameters
        circuit = Circuit(n)

        if encoding == "angle":
            rotation = encoded.metadata.get("rotation", "ry")
            gate_fn = {"ry": circuit.Ry, "rx": circuit.Rx, "rz": circuit.Rz}.get(rotation)
            if gate_fn is None:
                raise ValueError(f"Unknown rotation '{rotation}'.")
            for i, angle in enumerate(params):
                gate_fn(float(angle) / math.pi, i)

        elif encoding == "entangled_angle":
            rotation = encoded.metadata.get("rotation", "ry")
            layers = encoded.metadata.get("layers", 1)
            cnot_pairs = encoded.metadata.get("cnot_pairs", [])
            gate_fn = {"ry": circuit.Ry, "rx": circuit.Rx, "rz": circuit.Rz}.get(rotation)
            if gate_fn is None:
                raise ValueError(f"Unknown rotation '{rotation}'.")
            for _ in range(layers):
                for i, angle in enumerate(params):
                    gate_fn(float(angle) / math.pi, i)
                for ctrl, tgt in cnot_pairs:
                    circuit.CX(ctrl, tgt)

        elif encoding == "basis":
            for i, bit in enumerate(params):
                if bit == 1.0:
                    circuit.X(i)

        elif encoding == "amplitude":
            raise NotImplementedError(
                "Amplitude encoding requires exponential-depth state preparation "
                "which pytket does not natively support as a simple gate. "
                "Use QiskitExporter for amplitude encoding."
            )

        elif encoding == "iqp":
            d = n
            reps = encoded.metadata.get("reps", 2)
            x = params[:d]
            pair_angles = params[d:]
            for _ in range(reps):
                for i in range(d):
                    circuit.H(i)
                for i in range(d):
                    circuit.Rz(float(x[i]) / math.pi, i)
                idx = 0
                for i in range(d):
                    for j in range(i + 1, d):
                        angle = float(pair_angles[idx])
                        circuit.CX(i, j)
                        circuit.Rz(angle / math.pi, j)
                        circuit.CX(i, j)
                        idx += 1

        elif encoding == "reupload":
            layers = encoded.metadata.get("layers", 3)
            rotation = encoded.metadata.get("rotation", "ry")
            gate_fn = {"ry": circuit.Ry, "rx": circuit.Rx, "rz": circuit.Rz}.get(rotation)
            if gate_fn is None:
                raise ValueError(f"Unknown rotation '{rotation}'.")
            for _ in range(layers):
                for i, angle in enumerate(params):
                    gate_fn(float(angle) / math.pi, i)

        elif encoding == "hamiltonian":
            trotter_steps = encoded.metadata.get("trotter_steps", 4)
            for _ in range(trotter_steps):
                for i, angle in enumerate(params):
                    circuit.Rz(float(angle) / math.pi, i)

        elif encoding == "zz_feature_map":
            reps = encoded.metadata.get("reps", 2)
            single_angles = encoded.metadata["single_angles"]
            pair_angles = encoded.metadata["pair_angles"]
            pairs = encoded.metadata["pairs"]
            for _ in range(reps):
                for i in range(n):
                    circuit.H(i)
                for i, angle in enumerate(single_angles):
                    circuit.Rz(float(angle) / math.pi, i)
                for (i, j), angle in zip(pairs, pair_angles):
                    circuit.CX(i, j)
                    circuit.Rz(float(angle) / math.pi, j)
                    circuit.CX(i, j)

        elif encoding == "pauli_feature_map":
            reps = encoded.metadata.get("reps", 2)
            single_terms = encoded.metadata.get("single_terms", {})
            pair_terms = encoded.metadata.get("pair_terms", {})
            _gate = {"X": circuit.Rx, "Y": circuit.Ry, "Z": circuit.Rz}
            for _ in range(reps):
                for i in range(n):
                    circuit.H(i)
                for pauli, angles in single_terms.items():
                    gate_fn = _gate[pauli]
                    for i, angle in enumerate(angles):
                        gate_fn(float(angle) / math.pi, i)
                for pauli, entries in pair_terms.items():
                    for i, j, angle in entries:
                        if pauli == "XX":
                            circuit.H(i)
                            circuit.H(j)
                        elif pauli == "YY":
                            circuit.Sdg(i)
                            circuit.Sdg(j)
                        circuit.CX(i, j)
                        circuit.Rz(float(angle) / math.pi, j)
                        circuit.CX(i, j)
                        if pauli == "XX":
                            circuit.H(i)
                            circuit.H(j)
                        elif pauli == "YY":
                            circuit.S(i)
                            circuit.S(j)

        elif encoding == "random_fourier":
            rotation = encoded.metadata.get("rotation", "ry")
            gate_fn = {"ry": circuit.Ry, "rx": circuit.Rx, "rz": circuit.Rz}.get(rotation)
            if gate_fn is None:
                raise ValueError(f"Unknown rotation '{rotation}'.")
            for i, angle in enumerate(params):
                gate_fn(float(angle) / math.pi, i)

        elif encoding == "tensor_product":
            ry_angles = encoded.metadata["ry_angles"]
            rz_angles = encoded.metadata["rz_angles"]
            for k in range(n):
                circuit.Ry(float(ry_angles[k]) / math.pi, k)
                circuit.Rz(float(rz_angles[k]) / math.pi, k)

        elif encoding == "qaoa_problem":
            p = encoded.metadata.get("p", 1)
            beta = encoded.metadata.get("beta", 0.39269908169872414)
            local_angles = encoded.metadata["local_angles"]
            coupling_angles = encoded.metadata["coupling_angles"]
            pairs = encoded.metadata.get("pairs", [])
            for i in range(n):
                circuit.H(i)
            for _ in range(p):
                for i in range(n):
                    circuit.Rz(2.0 * float(local_angles[i]) / math.pi, i)
                for k, (i, j) in enumerate(pairs):
                    angle = 2.0 * float(coupling_angles[k])
                    circuit.CX(i, j)
                    circuit.Rz(angle / math.pi, j)
                    circuit.CX(i, j)
                for i in range(n):
                    circuit.Rx(2.0 * float(beta) / math.pi, i)

        else:
            raise ValueError(
                f"Unknown encoding '{encoding}'. "
                "Supported: angle, entangled_angle, basis, iqp, reupload, hamiltonian, "
                "zz_feature_map, pauli_feature_map, random_fourier, tensor_product, qaoa_problem."
            )

        return circuit

    def export_batch(self, encoded_list: list) -> list:
        """
        Export a list of EncodedResults to pytket Circuits.

        Parameters
        ----------
        encoded_list : list of EncodedResult

        Returns
        -------
        list of pytket.Circuit
            One circuit per sample.
        """
        return [self.export(e) for e in encoded_list]
