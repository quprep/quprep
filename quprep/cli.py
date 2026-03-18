"""Command-line interface for QuPrep.

Usage
-----
    quprep convert dataset.csv --encoding angle --framework qiskit
    quprep recommend dataset.csv --task classification --qubits 8
    quprep --version
"""

from __future__ import annotations

import argparse
import sys

from quprep import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quprep",
        description="Quantum data preparation — convert classical datasets to circuit-ready format.",
    )
    parser.add_argument("--version", action="version", version=f"quprep {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    # quprep convert
    convert = subparsers.add_parser("convert", help="Convert a dataset to a quantum circuit.")
    convert.add_argument("source", help="Input file path (CSV, JSON, Parquet, etc.) or '-' for stdin.")
    convert.add_argument("--encoding", default="angle", choices=["angle", "amplitude", "basis", "iqp", "reupload", "hamiltonian"], help="Encoding method (default: angle).")
    convert.add_argument("--framework", default="qiskit", choices=["qiskit", "pennylane", "cirq", "tket", "qasm"], help="Export target (default: qiskit).")
    convert.add_argument("--output", "-o", default=None, help="Output file path. Prints to stdout if omitted.")

    # quprep recommend
    recommend = subparsers.add_parser("recommend", help="Recommend the best encoding for a dataset.")
    recommend.add_argument("source", help="Input file path.")
    recommend.add_argument("--task", default="classification", choices=["classification", "regression", "qaoa", "kernel", "simulation"])
    recommend.add_argument("--qubits", type=int, default=None, help="Maximum qubit budget.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "convert":
        print(f"[quprep] convert — coming in v0.1.0")
        return 0

    if args.command == "recommend":
        print(f"[quprep] recommend — coming in v0.2.0")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
