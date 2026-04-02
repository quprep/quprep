"""Export encoded data in IQM native circuit format.

Generates circuits in the IQM JSON interchange format, compatible with
IQM quantum computers and the ``iqm-client`` Python package.

IQM native gate set
-------------------
- PRX(angle_t, phase_t) — parametric rotation in the XY plane.
  Performs a rotation by ``2π * angle_t`` about the axis
  ``φ = 2π * phase_t``.  Correspondence to standard gates:

    Rx(θ) → PRX(θ / (2π), 0)
    Ry(θ) → PRX(θ / (2π), 0.25)    # phase_t = 1/4 → Y axis
    Rz(θ) → H · PRX(θ/(2π), 0) · H  (virtual Z via basis change)

- CZ — controlled-Z two-qubit gate (native on IQM hardware).

Qubit naming
------------
IQM uses string qubit identifiers (e.g. ``"QB1"``, ``"QB2"``).

Output format
-------------
Returns a plain Python ``dict`` matching the IQM circuit JSON schema::

    {
        "name": "circuit",
        "instructions": [
            {"name": "prx", "qubits": ["QB1"], "args": {"angle_t": 0.25, "phase_t": 0.0}},
            {"name": "cz",  "qubits": ["QB1", "QB2"], "args": {}},
            ...
        ]
    }

This dict can be serialised with ``json.dumps()`` or passed directly to
``iqm_client.Circuit.from_dict()`` when ``iqm-client`` is installed.

Supported encodings
-------------------
- angle           : Ry (PRX with phase_t=0.25) per qubit.
- entangled_angle : Ry layer + CZ entangling layer, repeated layers times.
- basis           : X gates (PRX(0.5, 0)) on qubits where the bit is 1.
- iqp             : H + virtual-Rz + CZ-Rz-CZ interactions, repeated reps times.
- zz_feature_map  : H + virtual-Rz single-qubit + CZ-Rz-CZ pairwise, repeated reps times.
- reupload        : Ry repeated `layers` times per qubit.
- hamiltonian     : virtual-Rz per qubit, repeated trotter_steps times.
- tensor_product  : Ry + virtual-Rz per qubit.
- amplitude       : not supported.

No additional dependencies required to generate the circuit dict.
To submit to IQM hardware: pip install quprep[iqm]
"""

from __future__ import annotations

import math


def _prx(qubit: str, angle_t: float, phase_t: float) -> dict:
    return {"name": "prx", "qubits": [qubit], "args": {"angle_t": angle_t, "phase_t": phase_t}}


def _cz(q0: str, q1: str) -> dict:
    return {"name": "cz", "qubits": [q0, q1], "args": {}}


def _virtual_rz(qubit: str, theta: float) -> list[dict]:
    """Decompose Rz(θ) = H · Rx(θ) · H using PRX gates.

    H = PRX(0.5, 0.25) up to global phase in IQM convention.
    """
    h_op = _prx(qubit, 0.5, 0.25)
    rx_op = _prx(qubit, theta / (2 * math.pi), 0.0)
    return [h_op, rx_op, h_op]


def _ry(qubit: str, theta: float) -> dict:
    return _prx(qubit, theta / (2 * math.pi), 0.25)


def _rx(qubit: str, theta: float) -> dict:
    return _prx(qubit, theta / (2 * math.pi), 0.0)


def _x(qubit: str) -> dict:
    return _prx(qubit, 0.5, 0.0)


def _h(qubit: str) -> dict:
    return _prx(qubit, 0.5, 0.25)


class IQMExporter:
    """
    Export EncodedResult objects to IQM native circuit format (dict).

    Returns a Python ``dict`` matching the IQM circuit JSON schema.
    No SDK dependencies required — install ``quprep[iqm]`` only when
    you need to submit directly to IQM hardware via ``iqm-client``.

    Parameters
    ----------
    circuit_name : str
        Name field in the output circuit dict. Default ``"quprep_circuit"``.
    qubit_prefix : str
        Prefix for qubit labels. Default ``"QB"`` → ``"QB1"``, ``"QB2"``, ...
    """

    def __init__(self, circuit_name: str = "quprep_circuit", qubit_prefix: str = "QB"):
        self.circuit_name = circuit_name
        self.qubit_prefix = qubit_prefix

    def _qname(self, i: int) -> str:
        return f"{self.qubit_prefix}{i + 1}"

    def export(self, encoded) -> dict:
        """
        Convert an EncodedResult to an IQM circuit dict.

        Parameters
        ----------
        encoded : EncodedResult
            Output from any QuPrep encoder (except amplitude).

        Returns
        -------
        dict
            IQM circuit in JSON-serialisable dict form.
        """
        encoding = encoded.metadata.get("encoding", "unknown")
        params = encoded.parameters
        n = encoded.metadata.get("n_qubits", len(params))
        qubits = [self._qname(i) for i in range(n)]

        instructions = self._build_instructions(encoding, params, n, qubits, encoded.metadata)

        return {
            "name": self.circuit_name,
            "instructions": instructions,
        }

    def _build_instructions(
        self, encoding: str, params, n: int, qubits: list[str], meta: dict
    ) -> list[dict]:
        ops: list[dict] = []

        if encoding == "amplitude":
            raise NotImplementedError(
                "Amplitude encoding is not supported by IQMExporter."
            )

        if encoding == "angle":
            rotation = meta.get("rotation", "ry")
            for i, angle in enumerate(params):
                if rotation == "ry":
                    ops.append(_ry(qubits[i], float(angle)))
                elif rotation == "rx":
                    ops.append(_rx(qubits[i], float(angle)))
                else:  # rz via virtual
                    ops.extend(_virtual_rz(qubits[i], float(angle)))
            return ops

        if encoding == "entangled_angle":
            rotation = meta.get("rotation", "ry")
            layers = meta.get("layers", 1)
            cnot_pairs = meta.get("cnot_pairs", [])
            for _ in range(layers):
                for i, angle in enumerate(params):
                    if rotation == "ry":
                        ops.append(_ry(qubits[i], float(angle)))
                    elif rotation == "rx":
                        ops.append(_rx(qubits[i], float(angle)))
                    else:
                        ops.extend(_virtual_rz(qubits[i], float(angle)))
                for ctrl, tgt in cnot_pairs:
                    # CNOT = CZ with H on target before and after
                    ops.append(_h(qubits[tgt]))
                    ops.append(_cz(qubits[ctrl], qubits[tgt]))
                    ops.append(_h(qubits[tgt]))
            return ops

        if encoding == "basis":
            for i, bit in enumerate(params):
                if bit == 1.0:
                    ops.append(_x(qubits[i]))
            return ops

        if encoding == "iqp":
            d = n
            reps = meta.get("reps", 2)
            x = params[:d]
            pair_angles = params[d:]
            for _ in range(reps):
                for i in range(d):
                    ops.append(_h(qubits[i]))
                for i in range(d):
                    ops.extend(_virtual_rz(qubits[i], float(x[i])))
                idx = 0
                for i in range(d):
                    for j in range(i + 1, d):
                        angle = float(pair_angles[idx])
                        ops.append(_h(qubits[j]))
                        ops.append(_cz(qubits[i], qubits[j]))
                        ops.extend(_virtual_rz(qubits[j], angle))
                        ops.append(_cz(qubits[i], qubits[j]))
                        ops.append(_h(qubits[j]))
                        idx += 1
            return ops

        if encoding == "zz_feature_map":
            reps = meta.get("reps", 2)
            single_angles = meta["single_angles"]
            pair_angles = meta["pair_angles"]
            pairs = meta["pairs"]
            for _ in range(reps):
                for i in range(n):
                    ops.append(_h(qubits[i]))
                for i, angle in enumerate(single_angles):
                    ops.extend(_virtual_rz(qubits[i], float(angle)))
                for (i, j), angle in zip(pairs, pair_angles):
                    ops.append(_h(qubits[j]))
                    ops.append(_cz(qubits[i], qubits[j]))
                    ops.extend(_virtual_rz(qubits[j], float(angle)))
                    ops.append(_cz(qubits[i], qubits[j]))
                    ops.append(_h(qubits[j]))
            return ops

        if encoding == "tensor_product":
            ry_angles = meta["ry_angles"]
            rz_angles = meta["rz_angles"]
            for k in range(n):
                ops.append(_ry(qubits[k], float(ry_angles[k])))
                ops.extend(_virtual_rz(qubits[k], float(rz_angles[k])))
            return ops

        if encoding == "pauli_feature_map":
            reps = meta.get("reps", 2)
            single_terms = meta.get("single_terms", {})
            pair_terms = meta.get("pair_terms", {})
            for _ in range(reps):
                for i in range(n):
                    ops.append(_h(qubits[i]))
                for pauli, angles in single_terms.items():
                    for i, angle in enumerate(angles):
                        if pauli == "Z":
                            ops.extend(_virtual_rz(qubits[i], float(angle)))
                        elif pauli == "X":
                            ops.append(_rx(qubits[i], float(angle)))
                        else:
                            ops.append(_ry(qubits[i], float(angle)))
                for _pauli, entries in pair_terms.items():
                    for i, j, angle in entries:
                        ops.append(_h(qubits[j]))
                        ops.append(_cz(qubits[i], qubits[j]))
                        ops.extend(_virtual_rz(qubits[j], float(angle)))
                        ops.append(_cz(qubits[i], qubits[j]))
                        ops.append(_h(qubits[j]))
            return ops

        if encoding == "random_fourier":
            # Output angles are in [0,π] — encode as Ry rotations
            for i, angle in enumerate(params):
                ops.append(_ry(qubits[i], float(angle)))
            return ops

        if encoding == "reupload":
            rotation = meta.get("rotation", "ry")
            layers = meta.get("layers", 3)
            for _ in range(layers):
                for i, angle in enumerate(params):
                    if rotation == "ry":
                        ops.append(_ry(qubits[i], float(angle)))
                    elif rotation == "rx":
                        ops.append(_rx(qubits[i], float(angle)))
                    else:
                        ops.extend(_virtual_rz(qubits[i], float(angle)))
            return ops

        if encoding == "hamiltonian":
            trotter_steps = meta.get("trotter_steps", 4)
            for _ in range(trotter_steps):
                for i, angle in enumerate(params):
                    ops.extend(_virtual_rz(qubits[i], float(angle)))
            return ops

        if encoding == "qaoa_problem":
            p = meta.get("p", 1)
            beta = meta.get("beta", 0.39269908169872414)
            local_angles = meta["local_angles"]
            coupling_angles = meta["coupling_angles"]
            pairs = meta.get("pairs", [])
            for i in range(n):
                ops.append(_h(qubits[i]))
            for _ in range(p):
                for i in range(n):
                    ops.extend(_virtual_rz(qubits[i], 2.0 * float(local_angles[i])))
                for k, (i, j) in enumerate(pairs):
                    angle = 2.0 * float(coupling_angles[k])
                    # IQM uses CZ: CNOT = H(j) CZ H(j)
                    ops.append(_h(qubits[j]))
                    ops.append(_cz(qubits[i], qubits[j]))
                    ops.extend(_virtual_rz(qubits[j], angle))
                    ops.append(_cz(qubits[i], qubits[j]))
                    ops.append(_h(qubits[j]))
                for i in range(n):
                    ops.append(_rx(qubits[i], 2.0 * float(beta)))
            return ops

        raise ValueError(
            f"Unknown encoding '{encoding}'. "
            "Supported: angle, entangled_angle, basis, iqp, zz_feature_map, "
            "pauli_feature_map, random_fourier, tensor_product, qaoa_problem, "
            "reupload, hamiltonian."
        )

    def export_batch(self, encoded_list: list) -> list[dict]:
        """
        Export a list of EncodedResults to IQM circuit dicts.

        Parameters
        ----------
        encoded_list : list of EncodedResult

        Returns
        -------
        list of dict
        """
        return [self.export(e) for e in encoded_list]
