"""QUBO / Ising visualization utilities.

Requires matplotlib (pip install quprep[viz]).
"""

from __future__ import annotations

import numpy as np


def draw_qubo(qubo, title: str = "QUBO Matrix", cmap: str = "RdBu_r", ax=None):
    """
    Draw a heatmap of the QUBO Q matrix.

    Positive entries (coupling penalties) are shown in one colour,
    negative entries (incentives) in another. The diagonal encodes
    linear terms; off-diagonal entries encode quadratic interactions.

    Parameters
    ----------
    qubo : QUBOResult
        The QUBO problem to visualize.
    title : str
        Plot title. Default is "QUBO Matrix".
    cmap : str
        Matplotlib colormap. Default is "RdBu_r" (blue=negative, red=positive).
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. Creates a new figure if None.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ImportError
        If matplotlib is not installed. Install with: pip install quprep[viz]
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for draw_qubo(). "
            "Install with: pip install quprep[viz]"
        ) from exc

    Q = qubo.Q
    n = Q.shape[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, n * 0.6), max(4, n * 0.6)))

    # Use symmetric colour scale centred on zero
    vmax = np.max(np.abs(Q))
    vmax = vmax if vmax > 0 else 1.0

    im = ax.imshow(Q, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    # Colour bar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Coefficient value")

    # Labels
    if n <= 20:
        var_map_inv = {v: k for k, v in qubo.variable_map.items()} if qubo.variable_map else {}
        labels = [var_map_inv.get(i, f"x{i}") for i in range(n)]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)

    # Annotate cells for small matrices
    if n <= 10:
        for i in range(n):
            for j in range(n):
                val = Q[i, j]
                if abs(val) > 1e-10:
                    colour = "white" if abs(val) > vmax * 0.6 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color=colour)

    ax.set_title(title)
    ax.set_xlabel("Variable index")
    ax.set_ylabel("Variable index")

    return ax


def draw_ising(ising, title: str = "Ising Model", ax=None):
    """
    Draw the Ising model as a weighted graph.

    Nodes are arranged in a circle. Node colour represents the bias (h_i):
    blue = negative bias (prefers s=-1), red = positive (prefers s=+1).
    Edge colour and thickness represent coupling strength (J_ij).

    Parameters
    ----------
    ising : IsingResult
        The Ising model to visualize.
    title : str
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Existing axes. Creates a new figure if None.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for draw_ising(). "
            "Install with: pip install quprep[viz]"
        ) from exc

    h = ising.h
    J = ising.J
    n = len(h)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(5, n * 0.8), max(5, n * 0.8)))

    # Node positions — evenly spaced on a circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = np.column_stack([np.cos(angles), np.sin(angles)])

    # Draw edges (couplings J[i,j])
    j_vals = [J[i, j] for i in range(n) for j in range(i + 1, n) if abs(J[i, j]) > 1e-10]
    j_max = max(abs(v) for v in j_vals) if j_vals else 1.0
    cmap_edge = matplotlib.colormaps.get_cmap("coolwarm")

    for i in range(n):
        for j in range(i + 1, n):
            val = J[i, j]
            if abs(val) < 1e-10:
                continue
            lw = 1.0 + 4.0 * abs(val) / j_max
            colour = cmap_edge(0.5 + 0.5 * val / j_max)
            ax.plot(
                [pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                color=colour, linewidth=lw, alpha=0.7, zorder=1,
            )
            # Edge label at midpoint
            mx, my = (pos[i, 0] + pos[j, 0]) / 2, (pos[i, 1] + pos[j, 1]) / 2
            ax.text(mx, my, f"{val:.2f}", fontsize=6, ha="center", va="center",
                    color="gray", zorder=3)

    # Draw nodes (biases h[i])
    h_max = max(abs(v) for v in h) if np.any(np.abs(h) > 1e-10) else 1.0
    cmap_node = matplotlib.colormaps.get_cmap("RdBu_r")
    node_radius = 0.12

    for i in range(n):
        colour = cmap_node(0.5 + 0.5 * h[i] / h_max)
        circle = plt.Circle(pos[i], node_radius, color=colour, zorder=4, ec="black", lw=1)
        ax.add_patch(circle)
        ax.text(pos[i, 0], pos[i, 1], str(i), ha="center", va="center",
                fontsize=8, fontweight="bold", zorder=5)
        # h label outside node
        offset = 1.25 * node_radius
        ax.text(pos[i, 0] * (1 + offset), pos[i, 1] * (1 + offset),
                f"h={h[i]:.2f}", fontsize=7, ha="center", va="center", zorder=5)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)

    return ax
