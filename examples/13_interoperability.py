"""
13 — Interoperability & CLI Tools
==================================
Data connectors, dataset inspection, encoder benchmarking,
and reproducibility fingerprinting.

QuPrep v0.8.0 adds three data connectors (HuggingFace, OpenML, Kaggle),
two new CLI tools (inspect, benchmark), and pipeline reproducibility
fingerprinting.

    uv run python examples/13_interoperability.py

Optional deps used in this example:
    pip install quprep[openml]      # OpenMLIngester
    pip install quprep[huggingface] # HuggingFaceIngester
    pip install quprep[kaggle]      # KaggleIngester
"""

import json

import numpy as np

import quprep as qd
from quprep.ingest.numpy_ingester import NumpyIngester
from quprep.ingest.profiler import profile

rng = np.random.default_rng(42)

# ── 1. OpenML connector ───────────────────────────────────────────────────────
#
#   OpenMLIngester wraps the OpenML API. Load by integer ID or dataset name.
#   No account required for public datasets.

print("=" * 55)
print("OpenML connector")
print("=" * 55)

try:
    from quprep.ingest.openml_ingester import OpenMLIngester

    dataset = OpenMLIngester().load("iris")
    print(f"Dataset      : {dataset.metadata.get('name', 'iris')}")
    print(f"Shape        : {dataset.data.shape}")
    print(f"Features     : {dataset.feature_names}")
    print()
except ImportError:
    print("skipped — install quprep[openml] to run this section")
    print()
except Exception as exc:
    print(f"skipped — {exc}")
    print()

# ── 2. HuggingFace connector ──────────────────────────────────────────────────
#
#   HuggingFaceIngester supports auto-detection of the dataset modality:
#   tabular, image, text, or graph. Pass modality="auto" to let it sniff
#   the HuggingFace feature schema and dispatch accordingly.

print("=" * 55)
print("HuggingFace connector (tabular example)")
print("=" * 55)

try:
    from quprep.ingest.huggingface_ingester import HuggingFaceIngester

    # Load a small public tabular dataset from HuggingFace Hub
    ingester = HuggingFaceIngester(modality="auto", split="train")
    dataset = ingester.load("scikit-learn/iris")
    print(f"Shape        : {dataset.data.shape}")
    print(f"Modality     : {dataset.metadata.get('modality', 'tabular')}")
    print()
except ImportError:
    print("skipped — install quprep[huggingface] to run this section")
    print()
except Exception as exc:
    print(f"skipped — {exc}")
    print()

# ── 3. Dataset inspection (Python API) ───────────────────────────────────────
#
#   The same profile information shown by `quprep inspect` is available
#   programmatically via the profiler module.

print("=" * 55)
print("Dataset inspection — profiler API")
print("=" * 55)

X = rng.standard_normal((200, 6))
X[0, 2] = np.nan   # inject one missing value
dataset = NumpyIngester().load(X)

p = profile(dataset)
print(f"Shape        : {p.n_samples} samples × {p.n_features} features")
print(f"Missing      : {int(p.missing_counts.sum())} value(s)")
print(f"Sparsity     : {100.0 * (X == 0).sum() / X.size:.1f}% zeros")
print(f"Feature[0]   : [{p.mins[0]:.3f}, {p.maxs[0]:.3f}]  "
      f"mean={p.means[0]:.3f}  std={p.stds[0]:.3f}")
print()

# ── 4. Encoder benchmarking (Python API) ──────────────────────────────────────
#
#   `qd.compare_encodings()` runs multiple encoders and returns side-by-side
#   cost metrics — this is the programmatic equivalent of `quprep benchmark`.

print("=" * 55)
print("Encoder benchmarking — compare_encodings()")
print("=" * 55)

X_clean = rng.uniform(0, 1, (50, 4))
result = qd.compare_encodings(X_clean, include=["angle", "amplitude", "basis"])
print(str(result))

# ── 5. Reproducibility fingerprinting ────────────────────────────────────────
#
#   fingerprint_pipeline() produces a deterministic SHA-256 hash of the full
#   pipeline configuration (stage classes + parameters + dependency versions).
#   Use the hash in paper methods sections or experiment logs.

print("=" * 55)
print("Reproducibility fingerprinting")
print("=" * 55)

pipeline = qd.Pipeline(
    encoder=qd.AngleEncoder(rotation="ry"),
    reducer=qd.PCAReducer(n_components=3),
)
pipeline.fit(X_clean)

fp = qd.fingerprint_pipeline(pipeline)
print(f"Hash         : sha256:{fp.hash[:16]}...")
print(f"Stages       : {list(fp.config.get('stages', {}).keys())}")

# Same pipeline config → same hash
pipeline2 = qd.Pipeline(
    encoder=qd.AngleEncoder(rotation="ry"),
    reducer=qd.PCAReducer(n_components=3),
)
pipeline2.fit(X_clean)
fp2 = qd.fingerprint_pipeline(pipeline2)
print(f"Reproducible : {fp.hash == fp2.hash}")

# Serialise to JSON for logging
fp_json = json.loads(fp.to_json())
print(f"JSON keys    : {list(fp_json.keys())}")
print()

print("Example 13 complete.")
