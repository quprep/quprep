"""Circuit visualization — matplotlib and ASCII diagrams."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Column builder — encoding-agnostic gate sequence
# ---------------------------------------------------------------------------

_CNOT_W = 9  # fixed width for 2-qubit columns: "────●────"


def _build_columns(encoded) -> list[dict]:
    """Return a flat list of column dicts describing the circuit."""
    m = encoded.metadata
    enc = m.get("encoding", "")
    x = encoded.parameters
    n = m.get("n_qubits", 1)
    rot = m.get("rotation", "ry").upper()
    columns: list[dict] = []

    if enc == "angle":
        gates = {i: f"{rot}({x[i]:.2f})" for i in range(n)}
        columns.append({"type": "layer", "gates": gates})

    elif enc == "entangled_angle":
        layers = m.get("layers", 1)
        cnot_pairs = m.get("cnot_pairs", [])
        for _ in range(layers):
            gates = {i: f"{rot}({x[i]:.2f})" for i in range(n)}
            columns.append({"type": "layer", "gates": gates})
            for ctrl, tgt in cnot_pairs:
                columns.append({"type": "cnot", "ctrl": ctrl, "tgt": tgt})

    elif enc == "basis":
        # parameters are already binarized (0.0 / 1.0)
        gates = {i: "X" for i in range(n) if x[i] >= 0.5}
        columns.append({"type": "layer", "gates": gates})

    elif enc == "amplitude":
        gates = {i: "SP" for i in range(n)}
        columns.append({"type": "layer", "gates": gates})

    elif enc == "iqp":
        reps = m.get("reps", 1)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        pair_angles = x[n:]  # pairwise products stored after raw features
        for _ in range(reps):
            columns.append({"type": "layer", "gates": {i: "H" for i in range(n)}})
            for k, (i, j) in enumerate(pairs):
                angle = float(pair_angles[k]) if k < len(pair_angles) else 0.0
                columns.append({"type": "zz", "ctrl": i, "tgt": j, "angle": f"{angle:.2f}"})

    elif enc == "reupload":
        layers = m.get("layers", 1)
        for _ in range(layers):
            gates = {i: f"{rot}({x[i]:.2f})" for i in range(n)}
            columns.append({"type": "layer", "gates": gates})

    elif enc == "hamiltonian":
        steps = m.get("trotter_steps", 1)
        t = m.get("evolution_time", 1.0)
        dt = t / steps
        for _ in range(steps):
            gates = {i: f"Rz({x[i] * 2 * dt:.2f})" for i in range(n)}
            columns.append({"type": "layer", "gates": gates})

    else:
        # Unknown encoding — show raw parameters
        gates = {i: f"P({x[i]:.2f})" if i < len(x) else "─" for i in range(n)}
        columns.append({"type": "layer", "gates": gates})

    return columns


# ---------------------------------------------------------------------------
# ASCII rendering helpers
# ---------------------------------------------------------------------------


def _render_layer(gates: dict, n: int) -> list[str]:
    """Render a layer column into 2n-1 row strings."""
    if gates:
        max_lw = max(len(v) for v in gates.values())
    else:
        max_lw = 1
    total = max_lw + 6  # "──[" + label + "]──"
    rows: list[str] = []
    for i in range(n):
        if i in gates:
            label = gates[i].center(max_lw)
            rows.append(f"──[{label}]──")
        else:
            rows.append("─" * total)
        if i < n - 1:
            rows.append(" " * total)
    return rows


def _render_two_qubit(ctrl: int, tgt: int, n: int, sym_ctrl: str, sym_tgt: str) -> list[str]:
    """Render a 2-qubit column (CNOT or ZZ) into 2n-1 row strings."""
    lo, hi = min(ctrl, tgt), max(ctrl, tgt)
    half = _CNOT_W // 2
    wire = "─" * half
    space = " " * half
    rows: list[str] = []
    for i in range(n):
        if i == ctrl:
            rows.append(f"{wire}{sym_ctrl}{wire}")
        elif i == tgt:
            rows.append(f"{wire}{sym_tgt}{wire}")
        else:
            rows.append("─" * _CNOT_W)
        if i < n - 1:
            if lo <= i < hi:
                rows.append(f"{space}│{space}")
            else:
                rows.append(" " * _CNOT_W)
    return rows


# ---------------------------------------------------------------------------
# Public: ASCII
# ---------------------------------------------------------------------------


def draw_ascii(encoded, width: int = 80) -> str:  # noqa: ARG001
    """
    Return an ASCII circuit diagram for an EncodedResult.

    No additional dependencies required.

    Parameters
    ----------
    encoded : EncodedResult
    width : int
        Reserved for future use (target line width hint). Default 80.

    Returns
    -------
    str
        Multi-line ASCII string. Print with ``print(draw_ascii(encoded))``.
    """
    m = encoded.metadata
    n = m.get("n_qubits", 1)
    columns = _build_columns(encoded)

    num_rows = 2 * n - 1
    rows = [""] * num_rows

    # Qubit label prefix — right-aligned to widest label
    prefix_len = len(f"q[{n - 1}]") + 1
    for i in range(n):
        rows[2 * i] = f"q[{i}]".ljust(prefix_len)
    for i in range(n - 1):
        rows[2 * i + 1] = " " * prefix_len

    for col in columns:
        if col["type"] == "layer":
            col_rows = _render_layer(col["gates"], n)
        elif col["type"] == "cnot":
            col_rows = _render_two_qubit(col["ctrl"], col["tgt"], n, "●", "⊕")
        elif col["type"] == "zz":
            col_rows = _render_two_qubit(col["ctrl"], col["tgt"], n, "Z", "Z")
        else:
            col_rows = ["─" * _CNOT_W] * num_rows

        for r in range(num_rows):
            rows[r] += col_rows[r]

    # Trailing wire segment
    for i in range(n):
        rows[2 * i] += "──"

    enc = m.get("encoding", "unknown")
    depth = m.get("depth", "?")
    header = f"── {enc} | {n} qubits | depth {depth} ──"
    return header + "\n" + "\n".join(rows) + "\n"


# ---------------------------------------------------------------------------
# Matplotlib helpers
# ---------------------------------------------------------------------------


def _mpl_draw_gate(ax, x: float, y: float, label: str) -> None:
    import matplotlib.patches as mpatches  # noqa: PLC0415

    bw, bh = 0.72, 0.44
    rect = mpatches.FancyBboxPatch(
        (x - bw / 2, y - bh / 2),
        bw,
        bh,
        boxstyle="round,pad=0.04",
        facecolor="#E8F4FD",
        edgecolor="#2980B9",
        linewidth=1.5,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center", fontsize=7.5,
            fontfamily="monospace", zorder=4)


def _mpl_draw_cnot(ax, x: float, yc: float, yt: float) -> None:
    import matplotlib.patches as mpatches  # noqa: PLC0415

    ax.plot([x, x], [yc, yt], color="black", lw=1.5, zorder=2)
    # Control — filled circle
    ax.plot(x, yc, "o", color="black", markersize=9, zorder=4)
    # Target — open circle with cross (⊕)
    r = 0.19
    circle = mpatches.Circle((x, yt), r, color="white", ec="black", lw=1.5, zorder=3)
    ax.add_patch(circle)
    ax.plot([x - r, x + r], [yt, yt], color="black", lw=1.5, zorder=4)
    ax.plot([x, x], [yt - r, yt + r], color="black", lw=1.5, zorder=4)


def _mpl_draw_zz(ax, x: float, yc: float, yt: float) -> None:
    ax.plot([x, x], [yc, yt], color="#7D3C98", lw=1.5, linestyle="--", zorder=2)
    _mpl_draw_gate(ax, x, yc, "Z")
    _mpl_draw_gate(ax, x, yt, "Z")


# ---------------------------------------------------------------------------
# Public: matplotlib
# ---------------------------------------------------------------------------


def draw_matplotlib(encoded, filename: str | Path | None = None):
    """
    Draw a matplotlib circuit diagram.

    Requires: pip install quprep[viz]

    Parameters
    ----------
    encoded : EncodedResult
    filename : str or Path, optional
        Save to file if provided (PNG, PDF, SVG). Returns None.
        If None, returns the matplotlib Figure object.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is not installed. Run: pip install quprep[viz]"
        ) from None

    m = encoded.metadata
    n = m.get("n_qubits", 1)
    enc = m.get("encoding", "unknown")
    depth = m.get("depth", "?")
    columns = _build_columns(encoded)

    # Assign x-positions to columns
    col_x: list[float] = []
    x = 1.5
    for col in columns:
        col_x.append(x)
        x += 1.3 if col["type"] == "layer" else 0.85
    wire_end = x + 0.2

    # Figure sizing
    fig_w = max(wire_end + 0.6, 4.0)
    fig_h = max(n * 0.85 + 0.7, 2.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0.1, wire_end + 0.4)
    ax.set_ylim(-0.7, n - 0.3)
    ax.axis("off")

    wire_start = 1.0

    # y position: qubit 0 at top
    def yw(i: int) -> float:
        return float(n - 1 - i)

    # Qubit labels and horizontal wires
    for i in range(n):
        ax.text(
            0.85, yw(i), f"q[{i}]",
            ha="right", va="center", fontsize=9, fontfamily="monospace",
        )
        ax.plot([wire_start, wire_end], [yw(i), yw(i)], color="black", lw=1.5, zorder=1)

    # Draw each column
    for col, xc in zip(columns, col_x):
        if col["type"] == "layer":
            for qi, label in col["gates"].items():
                _mpl_draw_gate(ax, xc, yw(qi), label)
        elif col["type"] == "cnot":
            _mpl_draw_cnot(ax, xc, yw(col["ctrl"]), yw(col["tgt"]))
        elif col["type"] == "zz":
            _mpl_draw_zz(ax, xc, yw(col["ctrl"]), yw(col["tgt"]))

    ax.set_title(
        f"{enc} encoding  ·  {n} qubits  ·  depth {depth}",
        fontsize=10, pad=8,
    )

    if filename is not None:
        fig.savefig(Path(filename), bbox_inches="tight", dpi=150)
        plt.close(fig)
        return None
    return fig
