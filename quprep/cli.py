"""Command-line interface for QuPrep.

Usage
-----
    quprep convert dataset.csv --encoding angle --framework qasm
    quprep convert dataset.csv --encoding angle --framework qasm --output circuit.qasm
    quprep recommend dataset.csv --task classification --qubits 8
    quprep --version
"""

from __future__ import annotations

import argparse
import sys

from quprep import __version__

_PHASE1_ENCODINGS = ["angle", "amplitude", "basis"]
_PHASE2_ENCODINGS = ["iqp", "reupload", "hamiltonian"]
_ALL_ENCODINGS = _PHASE1_ENCODINGS + _PHASE2_ENCODINGS

_ALL_FRAMEWORKS = ["qasm", "qiskit", "pennylane", "cirq", "tket"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quprep",
        description=(
            "Quantum data preparation — convert classical datasets to circuit-ready format."
        ),
    )
    parser.add_argument("--version", action="version", version=f"quprep {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    # quprep convert
    convert = subparsers.add_parser(
        "convert",
        help="Convert a dataset to a quantum circuit.",
    )
    convert.add_argument("source", help="Input file path (CSV, TSV).")
    convert.add_argument(
        "--encoding", "-e",
        default="angle",
        choices=_ALL_ENCODINGS,
        help="Encoding method (default: angle).",
    )
    convert.add_argument(
        "--framework", "-f",
        default="qasm",
        choices=_ALL_FRAMEWORKS,
        help="Export target (default: qasm).",
    )
    convert.add_argument(
        "--rotation",
        default="ry",
        choices=["ry", "rx", "rz"],
        help="Rotation gate for angle encoding (default: ry).",
    )
    convert.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path. Prints to stdout if omitted.",
    )
    convert.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples to convert (default: all).",
    )

    # quprep recommend
    recommend = subparsers.add_parser(
        "recommend",
        help="Recommend the best encoding for a dataset.",
    )
    recommend.add_argument("source", help="Input file path.")
    recommend.add_argument(
        "--task",
        default="classification",
        choices=["classification", "regression", "qaoa", "kernel", "simulation"],
    )
    recommend.add_argument(
        "--qubits",
        type=int,
        default=None,
        help="Maximum qubit budget.",
    )

    return parser


def cmd_convert(args) -> int:
    try:
        import quprep
        result = quprep.prepare(
            args.source,
            encoding=args.encoding,
            framework=args.framework,
            rotation=args.rotation,
        )
    except FileNotFoundError:
        print(f"[quprep] File not found: {args.source}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"[quprep] Missing dependency: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[quprep] Error: {exc}", file=sys.stderr)
        return 1

    circuits = result.circuits or []
    if args.samples is not None:
        circuits = circuits[: args.samples]

    if args.framework == "qasm":
        output_lines = "\n".join(circuits)
        if args.output:
            from pathlib import Path
            Path(args.output).write_text(output_lines, encoding="utf-8")
            print(f"[quprep] Wrote {len(circuits)} circuit(s) to {args.output}")
        else:
            print(output_lines)
    else:
        # Non-QASM: print repr of each circuit object
        for i, circuit in enumerate(circuits):
            print(f"--- sample {i} ---")
            print(circuit)

    return 0


def cmd_recommend(args) -> int:
    try:
        from quprep.core.recommender import recommend
        rec = recommend(args.source, task=args.task, qubits=args.qubits)
    except FileNotFoundError:
        print(f"[quprep] File not found: {args.source}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"[quprep] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[quprep] Error: {exc}", file=sys.stderr)
        return 1

    print(str(rec))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "convert":
        return cmd_convert(args)

    if args.command == "recommend":
        return cmd_recommend(args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
