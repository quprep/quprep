"""
How to Choose an Encoder
=========================
QuPrep includes 15 encoders. This guide shows three ways to pick the right
one for your task: automatic recommendation, qubit budget estimation, and
side-by-side comparison.

    uv run python examples/how-to/choose_an_encoder.py
"""

import warnings

import numpy as np

import quprep as qd
from quprep import QuPrepWarning

rng = np.random.default_rng(42)
X = rng.uniform(0, 1, (80, 6))
y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
dataset = qd.NumpyIngester().load(X, y=y)

print(f"quprep {qd.__version__} | dataset: {X.shape}\n")


# ── 1. Automatic recommendation ───────────────────────────────────────────────

rec = qd.recommend(dataset, task="classification", qubits=6)

print("── 1. recommend() ───────────────────────────────────────────────────────")
print(f"   Top encoder : {rec.method}")
print(f"   Score       : {rec.score:.1f}")
print(f"   NISQ-safe   : {rec.nisq_safe}")
print(f"   Reason      : {rec.reason[:80]}...")
print()
print("   Alternatives:")
for alt in rec.alternatives[:3]:
    print(f"     {alt.method:<20} score={alt.score:.1f}  depth={alt.depth}")
print()


# ── 2. Qubit budget estimation ────────────────────────────────────────────────

sq = qd.suggest_qubits(dataset, task="classification", max_qubits=20)

print("── 2. suggest_qubits() ──────────────────────────────────────────────────")
print(f"   Suggested qubits : {sq.n_qubits}")
print(f"   NISQ-safe        : {sq.nisq_safe}")
print(f"   Encoding hint    : {sq.encoding_hint}")
print(f"   Reasoning        : {sq.reasoning[:80]}...")
print()


# ── 3. Side-by-side comparison ────────────────────────────────────────────────

with warnings.catch_warnings():
    warnings.simplefilter("ignore", QuPrepWarning)
    comparison = qd.compare_encodings(
        dataset,
        include=["angle", "amplitude", "basis", "iqp", "entangled_angle", "zz_feature_map"],
        task="classification",
        qubits=6,
    )

print("── 3. compare_encodings() ───────────────────────────────────────────────")
print(f"   {'Encoder':<20} {'Qubits':>6}  {'Depth':>6}  {'2Q gates':>8}  {'NISQ':>5}")
print("   " + "─" * 52)
for row in comparison.rows:
    nisq = "✓" if row.nisq_safe else "✗"
    print(f"   {row.encoding:<20} {row.n_qubits:>6}  {row.circuit_depth:>6}  "
          f"{row.two_qubit_gates:>8}  {nisq:>5}")
print(f"\n   Best overall: {comparison.best().encoding}")
print()


# ── 4. Build a pipeline from the suggestion ───────────────────────────────────
#
# suggest_pipeline() wraps recommend() and returns a PipelineSuggestion with
# a build() method. This is the most convenient path from recommendation to
# encoded circuits.

suggestion = qd.suggest_pipeline(dataset, task="classification", qubits=6)

print("── 4. suggest_pipeline().build() ────────────────────────────────────────")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", QuPrepWarning)
    result = suggestion.build().fit_transform(dataset)

print(f"   Encoder  : {result.encoded[0].metadata.get('encoding')}")
print(f"   Circuits : {len(result.encoded)}")
print(f"   Qubits   : {result.encoded[0].metadata.get('n_qubits')}")
