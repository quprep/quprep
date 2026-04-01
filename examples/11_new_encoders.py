"""
11 — New Encoders (v0.6.0)
===========================
ZZFeatureMap, PauliFeatureMap, RandomFourier, TensorProduct.

    uv run python examples/11_new_encoders.py
"""

import numpy as np

import quprep as qd
from quprep.encode.pauli_feature_map import PauliFeatureMapEncoder
from quprep.encode.random_fourier import RandomFourierEncoder
from quprep.encode.tensor_product import TensorProductEncoder
from quprep.encode.zz_feature_map import ZZFeatureMapEncoder

rng = np.random.default_rng(42)

exporter = qd.QASMExporter()

x3 = np.array([0.5, 1.2, 0.75]) * np.pi   # 3 features, scaled to [0, π]
x4 = np.array([0.3, 0.9, 0.6, 1.1]) * np.pi

# ── 1. ZZ Feature Map ─────────────────────────────────────────────────────────
#
#   Havlíček et al. 2019 (Qiskit convention).
#   H layer + single-qubit Rz(2(π−xᵢ)) + pairwise ZZ interactions.

print("=" * 55)
print("ZZFeatureMapEncoder  (reps=2)")
print("=" * 55)

enc_zz = ZZFeatureMapEncoder(reps=2)
result_zz = enc_zz.encode(x3)
print(f"n_qubits      : {result_zz.metadata['n_qubits']}")
print(f"reps          : {result_zz.metadata['reps']}")
print(f"pairs         : {result_zz.metadata['pairs']}")
print(f"single_angles : {[round(a, 4) for a in result_zz.metadata['single_angles']]}")
print(f"pair_angles   : {[round(a, 4) for a in result_zz.metadata['pair_angles']]}")
print()
print(exporter.export(result_zz))

# ── 2. Pauli Feature Map ──────────────────────────────────────────────────────
#
#   Generalized feature map with configurable Pauli strings.
#   Valid singles: X, Y, Z.  Valid pairs: XX, YY, ZZ, XZ, ZX, XY, YX, YZ, ZY.

print("=" * 55)
print("PauliFeatureMapEncoder  (paulis=[Z, ZZ], reps=1)")
print("=" * 55)

enc_p = PauliFeatureMapEncoder(paulis=["Z", "ZZ"], reps=1)
result_p = enc_p.encode(x3)
print(f"n_qubits      : {result_p.metadata['n_qubits']}")
st = {k: [round(v2, 4) for v2 in v] for k, v in result_p.metadata["single_terms"].items()}
print(f"single_terms  : {st}")
pair_terms_repr = {
    k: [(i, j, round(a, 4)) for i, j, a in v]
    for k, v in result_p.metadata["pair_terms"].items()
}
print(f"pair_terms    : {pair_terms_repr}")
print()
print(exporter.export(result_p))

# Mixed Paulis — higher expressivity
print("=" * 55)
print("PauliFeatureMapEncoder  (paulis=[Z, X, ZZ], reps=1)")
print("=" * 55)

enc_p2 = PauliFeatureMapEncoder(paulis=["Z", "X", "ZZ"], reps=1)
result_p2 = enc_p2.encode(x3)
print(f"single_terms  : { {k: len(v) for k, v in result_p2.metadata['single_terms'].items()} }")
print(f"pair_terms    : { {k: len(v) for k, v in result_p2.metadata['pair_terms'].items()} }")
print(f"depth         : {result_p2.metadata['depth']}")
print()

# ── 3. Random Fourier Features ────────────────────────────────────────────────
#
#   Approximates the RBF kernel via Bochner's theorem.
#   Requires fit() before encode(). n_components fixes qubit count.

print("=" * 55)
print("RandomFourierEncoder  (n_components=8, gamma=1.0)")
print("=" * 55)

X_train = rng.uniform(-1, 1, size=(100, 4))   # 100 training samples, 4 features
x_new   = rng.uniform(-1, 1, 4)

enc_rf = RandomFourierEncoder(n_components=8, gamma=1.0, random_state=42)
enc_rf.fit(X_train)                            # learns W and b
result_rf = enc_rf.encode(x_new)

print(f"Input dim     : {len(x_new)}  (4 features)")
print(f"n_qubits      : {result_rf.metadata['n_qubits']}  (always n_components)")
print(f"output range  : [{result_rf.parameters.min():.4f}, {result_rf.parameters.max():.4f}]")
print()
print(exporter.export(result_rf))

# Encode a batch
encoded_batch = [enc_rf.encode(X_train[i]) for i in range(3)]
print(f"Batch of 3 — first circuit qubit count: {encoded_batch[0].metadata['n_qubits']}")
print()

# ── 4. Tensor Product Encoding ────────────────────────────────────────────────
#
#   Ry + Rz per qubit. Encodes 2 features per qubit → ceil(d/2) qubits.
#   No entanglement. Odd-length inputs are zero-padded.

print("=" * 55)
print("TensorProductEncoder  (4 features → 2 qubits)")
print("=" * 55)

enc_tp = TensorProductEncoder()
result_tp = enc_tp.encode(x4)
print(f"Input dim     : {len(x4)}")
print(f"n_qubits      : {result_tp.metadata['n_qubits']}  (ceil(4/2) = 2)")
print(f"ry_angles     : {[round(a, 4) for a in result_tp.metadata['ry_angles']]}")
print(f"rz_angles     : {[round(a, 4) for a in result_tp.metadata['rz_angles']]}")
print()
print(exporter.export(result_tp))

# Odd-length input — zero-padded
x5 = np.array([0.3, 0.8, 1.2, 0.5, 0.9]) * np.pi
result_tp5 = enc_tp.encode(x5)
nq = result_tp5.metadata["n_qubits"]
print(f"5 features → n_qubits: {nq}  (ceil(5/2) = 3, last rz padded with 0)")
print()

# ── 5. Via prepare() ──────────────────────────────────────────────────────────

print("=" * 55)
print("prepare() — all new encoders")
print("=" * 55)

X = rng.uniform(0, 1, size=(10, 4))

for enc_name in ("zz_feature_map", "tensor_product"):
    result = qd.prepare(X, encoding=enc_name)
    meta = result.encoded[0].metadata
    print(f"  {enc_name:20s}  qubits={meta['n_qubits']}  depth={meta['depth']}")

# RandomFourier needs fit() — use the encoder directly in a pipeline
print()
enc_rf2 = RandomFourierEncoder(n_components=6, random_state=0)
enc_rf2.fit(X)
encoded_list = [enc_rf2.encode(X[i]) for i in range(len(X))]
print(f"  random_fourier        qubits={encoded_list[0].metadata['n_qubits']}  (6 components)")
