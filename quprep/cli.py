"""Command-line interface for QuPrep.

Usage
-----
    quprep convert dataset.csv --encoding angle --framework qasm
    quprep convert dataset.csv --encoding angle --save-dir ./circuits/
    quprep recommend dataset.csv --task classification --qubits 8
    quprep suggest dataset.csv --task classification
    quprep qubo maxcut --adjacency "0,1,1;1,0,1;1,1,0"
    quprep qubo knapsack --weights "2,3,4" --values "3,4,5" --capacity 5
    quprep qubo tsp --distances "0,1,2;1,0,1;2,1,0"
    quprep qubo schedule --times "3,1,4,2" --machines 2
    quprep inspect dataset.csv
    quprep inspect dataset.csv --task classification --qubits 8
    quprep benchmark dataset.csv
    quprep benchmark dataset.csv --samples 10 --task classification
    quprep benchmark dataset.csv --include angle,iqp,amplitude --output results.json
    quprep validate dataset.csv
    quprep validate dataset.csv --schema schema.json
    quprep compare dataset.csv --task classification --qubits 8
    quprep compare dataset.csv --include angle,iqp,amplitude
    quprep --version
"""

from __future__ import annotations

import argparse
import sys

from quprep import __version__

_ALL_ENCODINGS = [
    "angle", "entangled_angle", "amplitude", "basis", "iqp", "reupload", "hamiltonian",
    "zz_feature_map", "pauli_feature_map", "random_fourier", "tensor_product", "qaoa_problem",
]

_ALL_FRAMEWORKS = ["qasm", "qiskit", "pennylane", "cirq", "tket", "braket", "qsharp", "iqm"]


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
        "--save-dir",
        default=None,
        metavar="DIR",
        help=(
            "Save each sample as a separate QASM file in DIR "
            "(e.g. --save-dir ./circuits/). Only valid with --framework qasm."
        ),
    )
    convert.add_argument(
        "--stem",
        default="circuit",
        help="Filename stem for --save-dir output (default: circuit).",
    )
    convert.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples to convert (default: all).",
    )

    # quprep qubo
    qubo = subparsers.add_parser(
        "qubo",
        help="Build and optionally solve a QUBO problem.",
    )
    qubo_sub = qubo.add_subparsers(dest="qubo_command")

    # quprep qubo maxcut
    mc = qubo_sub.add_parser("maxcut", help="Max-Cut graph partitioning.")
    mc.add_argument(
        "--adjacency", "-a", required=True,
        help='Adjacency matrix as semicolon-separated rows, e.g. "0,1,1;1,0,1;1,1,0".',
    )
    mc.add_argument("--penalty", type=float, default=10.0)
    mc.add_argument("--solve", action="store_true", help="Brute-force solve (n <= 20).")

    # quprep qubo knapsack
    ks = qubo_sub.add_parser("knapsack", help="0/1 Knapsack.")
    ks.add_argument("--weights", "-w", required=True, help='Comma-separated weights, e.g. "2,3,4".')
    ks.add_argument("--values", "-v", required=True, help='Comma-separated values, e.g. "3,4,5".')
    ks.add_argument("--capacity", "-c", type=float, required=True)
    ks.add_argument("--penalty", type=float, default=None)
    ks.add_argument("--solve", action="store_true")

    # quprep qubo tsp
    tp = qubo_sub.add_parser("tsp", help="Travelling Salesman Problem.")
    tp.add_argument(
        "--distances", "-d", required=True,
        help='Distance matrix as semicolon-separated rows, e.g. "0,1,2;1,0,1;2,1,0".',
    )
    tp.add_argument("--penalty", type=float, default=None)
    tp.add_argument("--solve", action="store_true")

    # quprep qubo schedule
    sc = qubo_sub.add_parser("schedule", help="Job scheduling (load balancing).")
    sc.add_argument(
        "--times", "-t", required=True,
        help='Comma-separated processing times, e.g. "3,1,4,2".',
    )
    sc.add_argument("--machines", "-m", type=int, required=True)
    sc.add_argument("--penalty", type=float, default=None)
    sc.add_argument("--solve", action="store_true")

    # quprep qubo partition
    pt = qubo_sub.add_parser("partition", help="Number partitioning.")
    pt.add_argument(
        "--values", "-v", required=True,
        help='Comma-separated values, e.g. "3,1,1,2,2,1".',
    )
    pt.add_argument("--penalty", type=float, default=1.0)
    pt.add_argument("--solve", action="store_true")

    # quprep qubo portfolio
    pf = qubo_sub.add_parser("portfolio", help="Markowitz portfolio optimization.")
    pf.add_argument("--returns", "-r", required=True,
                    help='Comma-separated expected returns, e.g. "0.5,0.3,0.2,0.1".')
    pf.add_argument(
        "--covariance", required=True,
        help='Covariance matrix as semicolon-separated rows, e.g. "0.1,0.02;0.02,0.05".',
    )
    pf.add_argument("--budget", "-b", type=int, required=True,
                    help="Number of assets to select (budget constraint K).")
    pf.add_argument("--risk-penalty", type=float, default=1.0)
    pf.add_argument("--budget-penalty", type=float, default=None)
    pf.add_argument("--solve", action="store_true")

    # quprep qubo graphcolor
    gc = qubo_sub.add_parser("graphcolor", help="Graph colouring.")
    gc.add_argument("--adjacency", "-a", required=True,
                    help='Adjacency matrix as semicolon-separated rows, e.g. "0,1,1;1,0,1;1,1,0".')
    gc.add_argument("--colors", "-k", type=int, required=True,
                    help="Number of colours.")
    gc.add_argument("--penalty", type=float, default=10.0)
    gc.add_argument("--solve", action="store_true")

    # quprep qubo qaoa  — generate QAOA circuit
    qa = qubo_sub.add_parser("qaoa", help="Generate a QAOA circuit for a problem.")
    qa.add_argument("problem", choices=["maxcut", "knapsack", "tsp", "schedule", "partition"],
                    help="Problem type.")
    qa.add_argument("--adjacency", "-a", default=None)
    qa.add_argument("--weights", "-w", default=None)
    qa.add_argument("--values", "-v", default=None)
    qa.add_argument("--capacity", "-c", type=float, default=None)
    qa.add_argument("--distances", "-d", default=None)
    qa.add_argument("--times", "-t", default=None)
    qa.add_argument("--machines", type=int, default=None)
    qa.add_argument("--p", type=int, default=1, help="Number of QAOA layers.")
    qa.add_argument("--gamma", default=None, help='Comma-separated gamma values, e.g. "0.5,0.3".')
    qa.add_argument("--beta",  default=None, help='Comma-separated beta values,  e.g. "0.2,0.1".')
    qa.add_argument(
        "--output", "-o", default=None,
        help="Output file for QASM. Prints to stdout if omitted.",
    )

    # quprep qubo export  — serialize Q matrix
    ex = qubo_sub.add_parser("export", help="Export a QUBO Q matrix to JSON or numpy format.")
    ex.add_argument("problem", choices=["maxcut", "knapsack", "tsp", "schedule", "partition"])
    ex.add_argument("--adjacency", "-a", default=None)
    ex.add_argument("--weights", "-w", default=None)
    ex.add_argument("--values", "-v", default=None)
    ex.add_argument("--capacity", "-c", type=float, default=None)
    ex.add_argument("--distances", "-d", default=None)
    ex.add_argument("--times", "-t", default=None)
    ex.add_argument("--machines", type=int, default=None)
    ex.add_argument("--format", choices=["json", "npy"], default="json")
    ex.add_argument("--output", "-o", default=None, help="Output file path.")

    # quprep suggest
    suggest = subparsers.add_parser(
        "suggest",
        help="Suggest an appropriate qubit count for a dataset.",
    )
    suggest.add_argument("source", help="Input file path.")
    suggest.add_argument(
        "--task",
        default="classification",
        choices=["classification", "regression", "qaoa", "kernel", "simulation"],
        help="Target task (default: classification).",
    )
    suggest.add_argument(
        "--max-qubits",
        type=int,
        default=None,
        help="Hard upper bound on the qubit budget (default: 20).",
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

    # quprep compare
    compare = subparsers.add_parser(
        "compare",
        help="Compare all encoders on a dataset and show side-by-side cost stats.",
    )
    compare.add_argument("source", help="Input file path (CSV, TSV).")
    compare.add_argument(
        "--task",
        default=None,
        choices=["classification", "regression", "qaoa", "kernel", "simulation"],
        help="Highlight the recommended encoder for this task.",
    )
    compare.add_argument(
        "--qubits",
        type=int,
        default=None,
        help="Maximum qubit budget. Encoders exceeding it are flagged.",
    )
    compare.add_argument(
        "--include",
        default=None,
        metavar="ENCODINGS",
        help=(
            "Comma-separated list of encoders to include "
            "(e.g. --include angle,iqp,amplitude). Default: all."
        ),
    )
    compare.add_argument(
        "--exclude",
        default=None,
        metavar="ENCODINGS",
        help="Comma-separated list of encoders to exclude.",
    )

    # quprep inspect
    inspect = subparsers.add_parser(
        "inspect",
        help="Profile a dataset: shape, types, missing values, sparsity, and recommendation.",
    )
    inspect.add_argument("source", help="Input file path (CSV, TSV).")
    inspect.add_argument(
        "--task",
        default="classification",
        choices=["classification", "regression", "qaoa", "kernel", "simulation"],
        help="Task for encoding recommendation (default: classification).",
    )
    inspect.add_argument(
        "--qubits",
        type=int,
        default=None,
        help="Maximum qubit budget for recommendation.",
    )
    inspect.add_argument(
        "--no-recommend",
        action="store_true",
        help="Skip encoding recommendation.",
    )

    # quprep benchmark
    benchmark = subparsers.add_parser(
        "benchmark",
        help="Encode a dataset with all encoders and report gate count, depth, and timing.",
    )
    benchmark.add_argument("source", help="Input file path (CSV, TSV).")
    benchmark.add_argument(
        "--samples",
        type=int,
        default=5,
        metavar="N",
        help="Number of samples to encode per encoder (default: 5).",
    )
    benchmark.add_argument(
        "--task",
        default=None,
        choices=["classification", "regression", "qaoa", "kernel", "simulation"],
        help="Highlight the recommended encoder for this task.",
    )
    benchmark.add_argument(
        "--include",
        default=None,
        metavar="ENCODINGS",
        help="Comma-separated encoder names to include (default: all).",
    )
    benchmark.add_argument(
        "--exclude",
        default=None,
        metavar="ENCODINGS",
        help="Comma-separated encoder names to exclude.",
    )
    benchmark.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Save results as JSON to FILE.",
    )

    # quprep validate
    validate = subparsers.add_parser(
        "validate",
        help="Validate a dataset: report structure, NaN counts, and optional schema checks.",
    )
    validate.add_argument("source", help="Input CSV file path.")
    validate.add_argument(
        "--schema", "-s",
        default=None,
        metavar="SCHEMA_JSON",
        help=(
            "Path to a JSON schema file. Each entry must have 'name' and 'dtype' keys; "
            "'min_value', 'max_value', and 'nullable' are optional."
        ),
    )
    validate.add_argument(
        "--infer-schema",
        default=None,
        metavar="OUTPUT_JSON",
        help=(
            "Infer a schema from the dataset and write it to OUTPUT_JSON. "
            "Use '-' to print to stdout instead of saving to a file."
        ),
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
        if args.save_dir:
            from pathlib import Path

            from quprep.export.qasm_export import QASMExporter
            encoded = result.encoded or []
            if args.samples is not None:
                encoded = encoded[: args.samples]
            paths = QASMExporter().save_batch(encoded, Path(args.save_dir), stem=args.stem)
            print(f"[quprep] Wrote {len(paths)} circuit(s) to {args.save_dir}/")
        else:
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


def _parse_matrix(s: str):
    import numpy as np
    return np.array([[float(v) for v in row.split(",")] for row in s.split(";")])


def _parse_vec(s: str):
    import numpy as np
    return np.array([float(v) for v in s.split(",")])


def _print_qubo(result, solve: bool) -> int:
    import numpy as np
    print(f"Variables : {result.Q.shape[0]}")
    print(f"Offset    : {result.offset:.4f}")
    if result.n_original != result.Q.shape[0]:
        print(f"  ({result.n_original} original + {result.Q.shape[0] - result.n_original} slack)")
    print(f"Q matrix  :\n{np.array2string(result.Q, precision=4, suppress_small=True)}")
    if solve:
        from quprep.qubo.solver import solve_brute, solve_sa
        n = result.Q.shape[0]
        if n <= 20:
            sol = solve_brute(result)
            method = "exact"
        else:
            sol = solve_sa(result, n_steps=50_000, restarts=5, seed=0)
            method = "simulated annealing"
        bits = "".join(str(int(b)) for b in sol.x)
        print(f"\nSolver    : {method}")
        print(f"Best x    : {bits}")
        print(f"Energy    : {sol.energy:.6f}")
    return 0


def _build_qubo_from_args(problem: str, args):
    """Build a QUBOResult from CLI args for problem types used by qaoa/export."""
    if problem == "maxcut":
        from quprep.qubo.problems.maxcut import max_cut
        return max_cut(_parse_matrix(args.adjacency))
    if problem == "knapsack":
        from quprep.qubo.problems.knapsack import knapsack
        return knapsack(_parse_vec(args.weights), _parse_vec(args.values), args.capacity)
    if problem == "tsp":
        from quprep.qubo.problems.tsp import tsp
        return tsp(_parse_matrix(args.distances))
    if problem == "schedule":
        from quprep.qubo.problems.scheduling import scheduling
        return scheduling(_parse_vec(args.times), args.machines)
    if problem == "partition":
        from quprep.qubo.problems.number_partition import number_partition
        return number_partition(_parse_vec(args.values))
    raise ValueError(f"Unknown problem: {problem}")


def cmd_qubo(args) -> int:
    try:
        if args.qubo_command == "maxcut":
            from quprep.qubo.problems.maxcut import max_cut
            adj = _parse_matrix(args.adjacency)
            result = max_cut(adj)
            return _print_qubo(result, args.solve)

        if args.qubo_command == "knapsack":
            from quprep.qubo.problems.knapsack import knapsack
            w = _parse_vec(args.weights)
            v = _parse_vec(args.values)
            result = knapsack(w, v, args.capacity, penalty=args.penalty)
            return _print_qubo(result, args.solve)

        if args.qubo_command == "tsp":
            from quprep.qubo.problems.tsp import tsp
            D = _parse_matrix(args.distances)
            result = tsp(D, penalty=args.penalty)
            return _print_qubo(result, args.solve)

        if args.qubo_command == "schedule":
            from quprep.qubo.problems.scheduling import scheduling
            times = _parse_vec(args.times)
            result = scheduling(times, args.machines, penalty=args.penalty)
            return _print_qubo(result, args.solve)

        if args.qubo_command == "partition":
            from quprep.qubo.problems.number_partition import number_partition
            result = number_partition(_parse_vec(args.values), penalty=args.penalty)
            return _print_qubo(result, args.solve)

        if args.qubo_command == "portfolio":
            from quprep.qubo.problems.portfolio import portfolio
            returns = _parse_vec(args.returns)
            cov = _parse_matrix(args.covariance)
            result = portfolio(
                returns, cov, args.budget,
                risk_penalty=args.risk_penalty,
                budget_penalty=args.budget_penalty,
            )
            return _print_qubo(result, args.solve)

        if args.qubo_command == "graphcolor":
            from quprep.qubo.problems.graph_color import graph_color
            adj = _parse_matrix(args.adjacency)
            result = graph_color(adj, args.colors, penalty=args.penalty)
            return _print_qubo(result, args.solve)

        if args.qubo_command == "qaoa":
            from quprep.qubo.qaoa import qaoa_circuit
            result = _build_qubo_from_args(args.problem, args)
            gamma = [float(v) for v in args.gamma.split(",")] if args.gamma else None
            beta  = [float(v) for v in args.beta.split(",")]  if args.beta  else None
            qasm = qaoa_circuit(result, p=args.p, gamma=gamma, beta=beta)
            if args.output:
                from pathlib import Path
                Path(args.output).write_text(qasm, encoding="utf-8")
                print(f"[quprep] Wrote QAOA circuit to {args.output}")
            else:
                print(qasm)
            return 0

        if args.qubo_command == "export":
            import json
            result = _build_qubo_from_args(args.problem, args)
            fmt = args.format
            if fmt == "json":
                content = json.dumps(result.to_dict(), indent=2)
                if args.output:
                    from pathlib import Path
                    Path(args.output).write_text(content, encoding="utf-8")
                    print(f"[quprep] Wrote QUBO JSON to {args.output}")
                else:
                    print(content)
            else:  # npy
                import numpy as np
                out = args.output or "qubo.npy"
                np.save(out, result.Q)
                print(f"[quprep] Wrote Q matrix ({result.Q.shape}) to {out}")
            return 0

        # No subcommand — print help
        print("Usage: quprep qubo {maxcut,knapsack,tsp,schedule,partition,qaoa,export} [options]")
        return 0

    except Exception as exc:
        print(f"[quprep] Error: {exc}", file=__import__("sys").stderr)
        return 1


def cmd_inspect(args) -> int:
    try:
        from quprep.ingest.csv_ingester import CSVIngester
        dataset = CSVIngester().load(args.source)
    except FileNotFoundError:
        print(f"[quprep] File not found: {args.source}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[quprep] Failed to load '{args.source}': {exc}", file=sys.stderr)
        return 1


    from quprep.ingest.profiler import profile

    p = profile(dataset)

    # Header
    print(f"Source   : {args.source}")
    print(f"Shape    : {p.n_samples} samples × {p.n_features} features")

    if p.feature_names:
        preview = p.feature_names[:8]
        suffix = f" … (+{p.n_features - 8} more)" if p.n_features > 8 else ""
        print(f"Columns  : {', '.join(preview)}{suffix}")

    # Feature types
    if p.feature_types:
        type_counts: dict[str, int] = {}
        for t in p.feature_types:
            type_counts[t] = type_counts.get(t, 0) + 1
        print(f"Types    : {', '.join(f'{t}: {n}' for t, n in type_counts.items())}")

    # Missing values
    total_missing = int(p.missing_counts.sum()) if p.missing_counts is not None else 0
    if total_missing == 0:
        print("Missing  : none")
    else:
        total_values = p.n_samples * p.n_features
        print(
            f"Missing  : {total_missing} ({100.0 * total_missing / total_values:.1f}%"
            f" of {total_values} values)"
        )
        for i, count in enumerate(p.missing_counts):
            if count > 0:
                name = (
                    p.feature_names[i] if i < len(p.feature_names) else f"feature[{i}]"
                )
                print(f"           '{name}': {int(count)}")

    # Sparsity (fraction of zeros)
    total = dataset.data.size
    n_zeros = int((dataset.data == 0).sum())
    print(f"Sparsity : {100.0 * n_zeros / total:.1f}% zeros ({n_zeros}/{total})")

    # Per-feature stats (first 10)
    max_show = min(p.n_features, 10)
    print(f"\nFeature stats (first {max_show}):")
    col_w = max((len(n) for n in p.feature_names[:max_show]), default=12)
    for i in range(max_show):
        name = p.feature_names[i] if i < len(p.feature_names) else f"feature[{i}]"
        lo = p.mins[i] if p.mins is not None else float("nan")
        hi = p.maxs[i] if p.maxs is not None else float("nan")
        mu = p.means[i] if p.means is not None else float("nan")
        sd = p.stds[i] if p.stds is not None else float("nan")
        missing_n = int(p.missing_counts[i]) if p.missing_counts is not None else 0
        missing_str = f"  {missing_n} missing" if missing_n else ""
        print(
            f"  {name:<{col_w}}  [{lo:.4g}, {hi:.4g}]"
            f"  mean={mu:.4g}  std={sd:.4g}{missing_str}"
        )
    if p.n_features > max_show:
        print(f"  … ({p.n_features - max_show} more features)")

    # Encoding recommendation
    if not args.no_recommend:
        print()
        try:
            from quprep.core.recommender import recommend
            rec = recommend(dataset, task=args.task, qubits=args.qubits)
            print(str(rec))
        except Exception as exc:
            print(f"[quprep] Recommendation failed: {exc}", file=sys.stderr)

    return 0


def cmd_benchmark(args) -> int:
    import time


    # Load data
    try:
        from quprep.ingest.csv_ingester import CSVIngester
        dataset = CSVIngester().load(args.source)
    except FileNotFoundError:
        print(f"[quprep] File not found: {args.source}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[quprep] Failed to load '{args.source}': {exc}", file=sys.stderr)
        return 1

    from quprep.compare import _default_encoders
    from quprep.normalize.scalers import auto_normalizer
    from quprep.validation.cost import estimate_cost

    encoders = _default_encoders()

    # Filter by --include / --exclude
    include = [s.strip() for s in args.include.split(",")] if args.include else None
    exclude = {s.strip() for s in args.exclude.split(",")} if args.exclude else set()
    if include is not None:
        unknown = set(include) - set(encoders)
        if unknown:
            print(f"[quprep] Unknown encoder(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            return 1
        encoders = {k: v for k, v in encoders.items() if k in include}
    encoders = {k: v for k, v in encoders.items() if k not in exclude}

    # Determine recommended encoder (for task annotation)
    recommended: str | None = None
    if args.task:
        try:
            from quprep.core.recommender import recommend
            rec = recommend(dataset, task=args.task)
            recommended = rec.method
        except Exception:
            pass

    # Slice dataset to --samples rows
    n_bench = min(args.samples, dataset.n_samples)
    from quprep.core.dataset import Dataset
    bench_dataset = Dataset(
        data=dataset.data[:n_bench],
        feature_names=dataset.feature_names,
        feature_types=dataset.feature_types,
    )

    # Header
    print(f"Source   : {args.source}")
    print(
        f"Shape    : {dataset.n_samples} samples × {dataset.n_features} features"
        f"  (benchmarking on {n_bench} sample{'s' if n_bench != 1 else ''})"
    )
    print()

    # Run each encoder
    col_enc = 20
    col_q   = 7
    col_g   = 7
    col_d   = 7
    col_2q  = 9
    col_t   = 13
    col_n   = 9

    header = (
        f"{'Encoding':<{col_enc}}  {'Qubits':>{col_q}}  {'Gates':>{col_g}}"
        f"  {'Depth':>{col_d}}  {'2Q-Gates':>{col_2q}}"
        f"  {'Time/sample':>{col_t}}  {'NISQ':>{col_n}}"
    )
    sep = (
        f"{'-' * col_enc}  {'-' * col_q}  {'-' * col_g}"
        f"  {'-' * col_d}  {'-' * col_2q}"
        f"  {'-' * col_t}  {'-' * col_n}"
    )
    print(header)
    print(sep)

    results = []
    for name, encoder in encoders.items():
        # Normalize data the same way the pipeline would
        _encoding_to_key = {
            "angle": "angle_ry",
            "entangled_angle": "angle_ry",
            "amplitude": "amplitude",
            "basis": "basis",
            "iqp": "angle_ry",
            "reupload": "angle_ry",
            "hamiltonian": "hamiltonian",
            "qaoa_problem": "angle_ry",
        }
        norm_key = _encoding_to_key.get(name)
        try:
            normalizer = auto_normalizer(norm_key) if norm_key else None
            norm_dataset = normalizer.fit_transform(bench_dataset) if normalizer else bench_dataset
        except Exception:
            norm_dataset = bench_dataset

        # Time encoding
        error = None
        t_ms = float("nan")
        cost = None
        try:
            t0 = time.perf_counter()
            encoder.encode_batch(norm_dataset)
            t1 = time.perf_counter()
            t_ms = 1000.0 * (t1 - t0) / n_bench
            cost = estimate_cost(encoder, dataset.n_features)
        except Exception as exc:
            error = str(exc)

        marker = " *" if name == recommended else "  "
        label = name + marker

        if error:
            row_str = (
                f"{label:<{col_enc}}  {'—':>{col_q}}  {'—':>{col_g}}"
                f"  {'—':>{col_d}}  {'—':>{col_2q}}"
                f"  {'—':>{col_t}}  ERROR: {error[:30]}"
            )
        else:
            time_str = f"{t_ms:.2f} ms"
            nisq_str = "yes" if cost.nisq_safe else "NO"
            row_str = (
                f"{label:<{col_enc}}  {cost.n_qubits:>{col_q}}  {cost.gate_count:>{col_g}}"
                f"  {cost.circuit_depth:>{col_d}}  {cost.two_qubit_gates:>{col_2q}}"
                f"  {time_str:>{col_t}}  {nisq_str:>{col_n}}"
            )
            results.append({
                "encoding": name,
                "n_qubits": cost.n_qubits,
                "gate_count": cost.gate_count,
                "circuit_depth": cost.circuit_depth,
                "two_qubit_gates": cost.two_qubit_gates,
                "time_per_sample_ms": round(t_ms, 4),
                "nisq_safe": cost.nisq_safe,
                "warning": cost.warning,
            })

        print(row_str)

    if recommended:
        print(f"\n* recommended for task={args.task}")

    warnings = [r for r in results if r.get("warning")]
    if warnings:
        print()
        for r in warnings:
            print(f"  [{r['encoding']}] {r['warning']}")

    if args.output:
        import json
        from pathlib import Path
        payload = {
            "source": args.source,
            "n_samples": dataset.n_samples,
            "n_features": dataset.n_features,
            "n_bench_samples": n_bench,
            "task": args.task,
            "recommended": recommended,
            "results": results,
        }
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n[quprep] Results saved to {args.output}")

    return 0


def cmd_validate(args) -> int:
    try:
        from quprep.ingest.csv_ingester import CSVIngester
        dataset = CSVIngester().load(args.source)
    except FileNotFoundError:
        print(f"[quprep] File not found: {args.source}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[quprep] Failed to load '{args.source}': {exc}", file=sys.stderr)
        return 1

    import numpy as np

    print(f"Dataset : {args.source}")
    print(f"Shape   : {dataset.n_samples} samples × {dataset.n_features} features")
    if dataset.feature_names:
        names_preview = dataset.feature_names[:8]
        suffix = " ..." if dataset.n_features > 8 else ""
        print(f"Columns : {', '.join(names_preview)}{suffix}")

    # NaN report
    nan_cols = []
    for i in range(dataset.n_features):
        col = dataset.data[:, i]
        n_nan = int(np.isnan(col).sum())
        if n_nan:
            col_name = (
                dataset.feature_names[i]
                if i < len(dataset.feature_names)
                else f"feature[{i}]"
            )
            nan_cols.append((col_name, n_nan))
    if nan_cols:
        print(f"NaN     : {len(nan_cols)} column(s) with missing values")
        for col_name, n_nan in nan_cols:
            pct = 100.0 * n_nan / dataset.n_samples
            print(f"          '{col_name}': {n_nan} ({pct:.1f}%)")
    else:
        print("NaN     : none")

    # Value ranges
    print("Ranges  :")
    for i in range(min(dataset.n_features, 10)):
        col = dataset.data[:, i]
        valid = col[~np.isnan(col)]
        col_name = (
            dataset.feature_names[i]
            if i < len(dataset.feature_names)
            else f"feature[{i}]"
        )
        if valid.size > 0:
            print(f"          '{col_name}': [{valid.min():.4g}, {valid.max():.4g}]")
        else:
            print(f"          '{col_name}': all NaN")
    if dataset.n_features > 10:
        print(f"          ... ({dataset.n_features - 10} more columns)")

    # Schema inference
    if args.infer_schema:
        from quprep.validation.schema import DataSchema
        inferred = DataSchema.infer(dataset)
        json_str = inferred.to_json()
        if args.infer_schema == "-":
            print("\nInferred schema:")
            print(json_str)
        else:
            from pathlib import Path
            Path(args.infer_schema).write_text(json_str, encoding="utf-8")
            print(f"\nInferred schema written to {args.infer_schema}")

    # Schema validation
    if args.schema:
        import json
        try:
            with open(args.schema, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            print(f"\n[quprep] Cannot load schema file: {exc}", file=sys.stderr)
            return 1

        from quprep.validation.schema import DataSchema, FeatureSpec, SchemaViolationError
        specs = [
            FeatureSpec(
                name=entry["name"],
                dtype=entry["dtype"],
                min_value=entry.get("min_value"),
                max_value=entry.get("max_value"),
                nullable=entry.get("nullable", False),
            )
            for entry in raw
        ]
        schema = DataSchema(specs)
        print("\nSchema  : checking ...")
        try:
            schema.validate(dataset)
            print("Schema  : OK — no violations")
        except SchemaViolationError as exc:
            print(f"Schema  : FAILED\n{exc}", file=sys.stderr)
            return 1

    return 0


def cmd_suggest(args) -> int:
    try:
        from quprep.core.qubit_suggestion import suggest_qubits
        suggestion = suggest_qubits(
            args.source,
            task=args.task,
            max_qubits=args.max_qubits,
        )
    except FileNotFoundError:
        print(f"[quprep] File not found: {args.source}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"[quprep] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[quprep] Error: {exc}", file=sys.stderr)
        return 1

    print(str(suggestion))
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


def cmd_compare(args) -> int:
    include = [s.strip() for s in args.include.split(",")] if args.include else None
    exclude = [s.strip() for s in args.exclude.split(",")] if args.exclude else None
    try:
        from quprep.compare import compare_encodings
        result = compare_encodings(
            args.source,
            task=args.task,
            qubits=args.qubits,
            include=include,
            exclude=exclude,
        )
    except FileNotFoundError:
        print(f"[quprep] File not found: {args.source}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"[quprep] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[quprep] Error: {exc}", file=sys.stderr)
        return 1

    print(str(result))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "convert":
        return cmd_convert(args)

    if args.command == "qubo":
        return cmd_qubo(args)

    if args.command == "inspect":
        return cmd_inspect(args)

    if args.command == "benchmark":
        return cmd_benchmark(args)

    if args.command == "suggest":
        return cmd_suggest(args)

    if args.command == "recommend":
        return cmd_recommend(args)

    if args.command == "validate":
        return cmd_validate(args)

    if args.command == "compare":
        return cmd_compare(args)

    return 1  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
