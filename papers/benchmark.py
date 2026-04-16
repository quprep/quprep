"""QuPrep encoder benchmark — circuit depth and gate count across all encoders.

Runs all twelve classical-data encoders on the Iris and Heart Disease datasets
and prints a side-by-side comparison table of qubit count, circuit depth, and
single/two-qubit gate counts.  No quantum backend is required; all metrics are
derived analytically from the CostEstimate attached to each EncodedResult.

Usage
-----
    python benchmark.py

Requirements
------------
    pip install quprep scikit-learn
    # Heart Disease dataset is fetched from OpenML (ID 53) — internet access needed.
"""

from __future__ import annotations

import sys
import textwrap

import numpy as np

try:
    import quprep as qd
except ImportError:
    sys.exit("quprep is not installed. Run: pip install quprep")

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_iris() -> np.ndarray:
    from sklearn.datasets import load_iris as _load
    return _load(return_X_y=True)[0]


def load_heart() -> np.ndarray:
    """Heart Disease dataset from OpenML (ID 53). Falls back to Iris on failure."""
    try:
        ds = qd.OpenMLIngester().load(53)
        return ds.data
    except Exception as exc:
        print(f"  [warn] Could not fetch Heart Disease from OpenML ({exc}); using Iris.")
        return load_iris()


# ---------------------------------------------------------------------------
# Encoder catalogue
# ---------------------------------------------------------------------------

ENCODERS: list[tuple[str, str]] = [
    ("angle",            "Angle (Ry)"),
    ("amplitude",        "Amplitude"),
    ("basis",            "Basis"),
    ("iqp",              "IQP"),
    ("entangled_angle",  "Entangled Angle"),
    ("reupload",         "Data Re-upload"),
    ("hamiltonian",      "Hamiltonian"),
    ("zz_feature_map",   "ZZ Feature Map"),
    ("pauli_feature_map","Pauli Feature Map"),
    ("random_fourier",   "Random Fourier"),
    ("tensor_product",   "Tensor Product"),
    ("qaoa_problem",     "QAOA Problem"),
]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(X: np.ndarray, dataset_name: str) -> list[dict]:
    print(f"\nDataset: {dataset_name}  (shape {X.shape})")
    print("-" * 70)

    rows: list[dict] = []
    for enc_key, enc_label in ENCODERS:
        try:
            result = qd.prepare(X, encoding=enc_key, framework="qasm")
            cost   = result.cost          # CostEstimate

            # Parse basic metrics from CostEstimate
            n_qubits  = cost.n_qubits
            depth     = cost.circuit_depth
            n_gates   = cost.gate_count
            n_2q      = cost.two_qubit_gates
            nisq      = "Yes" if cost.nisq_safe else "No"

            rows.append({
                "Encoding":     enc_label,
                "Qubits":       n_qubits,
                "Depth":        depth,
                "Gates (total)":n_gates,
                "2-qubit gates":n_2q,
                "NISQ-safe":    nisq,
            })

        except Exception as exc:
            rows.append({
                "Encoding":     enc_label,
                "Qubits":       "—",
                "Depth":        "—",
                "Gates (total)":"—",
                "2-qubit gates":"—",
                "NISQ-safe":    f"Error: {exc}",
            })

    return rows


def print_table(rows: list[dict], dataset_name: str) -> None:
    cols    = ["Encoding", "Qubits", "Depth", "Gates (total)", "2-qubit gates", "NISQ-safe"]
    widths  = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    sep     = "  ".join("-" * widths[c] for c in cols)
    header  = "  ".join(c.ljust(widths[c]) for c in cols)

    print(f"\n{'='*len(sep)}")
    print(f"Results — {dataset_name}")
    print(f"{'='*len(sep)}")
    print(header)
    print(sep)
    for row in rows:
        print("  ".join(str(row[c]).ljust(widths[c]) for c in cols))
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("QuPrep Encoder Benchmark")
    print("=" * 70)
    print(textwrap.dedent("""\
        Measures circuit depth and gate count for all twelve classical-data
        encoders on two benchmark datasets.  Metrics are computed analytically
        via CostEstimate — no quantum simulator required.
    """))

    datasets = [
        ("Iris",          load_iris()),
        ("Heart Disease", load_heart()),
    ]

    all_results: dict[str, list[dict]] = {}
    for name, X in datasets:
        rows = run_benchmark(X, name)
        all_results[name] = rows
        print_table(rows, name)

    # Summary: per-encoder average depth across both datasets
    print("=" * 70)
    print("Average circuit depth across datasets")
    print("-" * 40)
    for enc_key, enc_label in ENCODERS:
        depths = []
        for name, _ in datasets:
            row = next((r for r in all_results[name] if r["Encoding"] == enc_label), None)
            if row and isinstance(row["Depth"], int):
                depths.append(row["Depth"])
        avg = f"{sum(depths)/len(depths):.1f}" if depths else "—"
        print(f"  {enc_label:<22}: {avg}")
    print()


if __name__ == "__main__":
    main()
