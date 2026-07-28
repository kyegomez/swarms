"""
Render benchmark results from ``graph_workflow_bench.py`` as PNG figures.

    python3 tests/benchmarks/graph_workflow_benchmarks/plot_results.py tests/benchmarks/graph_workflow_benchmarks/results/latest.json
    python3 tests/benchmarks/graph_workflow_benchmarks/plot_results.py tests/benchmarks/graph_workflow_benchmarks/results/after.json \\
        --baseline tests/benchmarks/graph_workflow_benchmarks/results/before.json

Figures are written to ``tests/benchmarks/graph_workflow_benchmarks/results/figures/`` in both light and dark
variants. Every figure is direct-labelled: three of the palette slots sit below
3:1 contrast on a light surface, so identity never rests on color alone.
"""

import argparse
import json
import os
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

PHASES = ["build", "compile", "first_run", "steady_run"]
PHASE_TITLES = {
    "build": "Build (add nodes + edges)",
    "compile": "Compile (topological pre-computation)",
    "first_run": "First run (build + compile + execute)",
    "steady_run": "Steady run (execute, already compiled)",
}

# Categorical slots 1-3 from the validated reference palette. The order is the
# CVD-safety mechanism, not cosmetic — do not reassign or cycle.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "text_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": {
            "swarms-networkx": "#2a78d6",
            "swarms-rustworkx": "#eb6834",
            "langgraph": "#1baf7a",
        },
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "text_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": {
            "swarms-networkx": "#3987e5",
            "swarms-rustworkx": "#d95926",
            "langgraph": "#199e70",
        },
    },
}

LABELS = {
    "swarms-networkx": "swarms (networkx)",
    "swarms-rustworkx": "swarms (rustworkx)",
    "langgraph": "LangGraph",
}


def ms_fmt(value: float, _pos: int = 0) -> str:
    """Axis tick formatter: drop trailing zeros, keep sub-millisecond legible."""
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.0f}"
    if value >= 1:
        return f"{value:.1f}"
    return f"{value:.2f}"


def style(theme: Dict[str, Any]) -> None:
    """Apply recessive chrome: hairline grid, muted ticks, no top/right spines."""
    plt.rcParams.update(
        {
            "figure.facecolor": theme["surface"],
            "axes.facecolor": theme["surface"],
            "savefig.facecolor": theme["surface"],
            "text.color": theme["text"],
            "axes.labelcolor": theme["text_secondary"],
            "axes.edgecolor": theme["axis"],
            "xtick.color": theme["muted"],
            "ytick.color": theme["muted"],
            "grid.color": theme["grid"],
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica Neue",
                "Helvetica",
                "Arial",
                "DejaVu Sans",
            ],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )


def _finish_axis(ax, theme: Dict[str, Any]) -> None:
    ax.grid(True, axis="y", linewidth=0.6, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.8)


def _direct_labels(ax, entries, min_gap: float = 0.055) -> None:
    """
    Place end-of-line labels, nudged apart so converging series stay readable.

    Labels are the accessibility channel here — three palette slots fall below
    3:1 on the light surface — so they must never collide into illegibility.
    Positions are converted to axes fraction, separated by ``min_gap``, and
    re-anchored there, which keeps the nudge independent of the data scale.

    Args:
        ax: The axes to annotate.
        entries: Iterable of ``(x, y, text, color)`` in data coordinates.
        min_gap (float): Minimum vertical separation, in axes fraction.
    """
    if not entries:
        return

    to_axes = ax.transAxes.inverted()
    points = []
    for x, y, text, color in entries:
        ax_x, ax_y = to_axes.transform(ax.transData.transform((x, y)))
        points.append([ax_x, ax_y, text, color])

    points.sort(key=lambda p: p[1])
    for i in range(1, len(points)):
        if points[i][1] - points[i - 1][1] < min_gap:
            points[i][1] = points[i - 1][1] + min_gap

    for ax_x, ax_y, text, color in points:
        ax.annotate(
            text,
            xy=(ax_x, ax_y),
            xycoords="axes fraction",
            xytext=(5, 0),
            textcoords="offset points",
            color=color,
            fontsize=7.5,
            va="center",
            zorder=4,
        )


# ----------------------------------------------------------------------------
# figure: scaling — how each phase grows with graph size
# ----------------------------------------------------------------------------
def fig_scaling(
    results: Dict[str, Any], topo: str, theme_name: str, out_dir: str
) -> str:
    """Four panels (one per phase) of median time vs node count, log-log."""
    theme = THEMES[theme_name]
    style(theme)

    sizes_map = results["topologies"][topo]
    sizes = sorted(sizes_map, key=int)
    impls = results["impls"]

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0))
    for ax, phase in zip(axes.flat, PHASES):
        label_entries = []
        for impl in impls:
            xs, ys = [], []
            for size in sizes:
                stats = sizes_map[size].get(impl, {}).get(phase)
                if stats:
                    xs.append(int(size))
                    ys.append(stats["median_ms"])
            if not xs:
                continue
            color = theme["series"][impl]
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=5,
                markeredgecolor=theme["surface"],
                markeredgewidth=1.2,
                zorder=3,
            )
            label_entries.append(
                (xs[-1], ys[-1], LABELS[impl], color)
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(PHASE_TITLES[phase], color=theme["text"], pad=8)
        ax.set_xlabel("nodes")
        ax.set_ylabel("median ms")
        ax.set_xticks([int(s) for s in sizes])
        ax.set_xticks([], minor=True)
        ax.get_xaxis().set_major_formatter(
            FuncFormatter(lambda v, p: f"{v:.0f}")
        )
        ax.get_yaxis().set_major_formatter(FuncFormatter(ms_fmt))
        ax.margins(x=0.30)
        _finish_axis(ax, theme)
        # Placed last: axes limits must be final before data→axes conversion.
        _direct_labels(ax, label_entries)

    fig.suptitle(
        f"GraphWorkflow orchestration overhead — {topo} topology",
        color=theme["text"],
        fontsize=12,
        y=0.985,
    )
    fig.text(
        0.5,
        0.005,
        "No-op nodes; lower is better. Log-log axes.",
        ha="center",
        color=theme["muted"],
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    return _save(fig, out_dir, f"scaling_{topo}", theme_name)


# ----------------------------------------------------------------------------
# figure: per-phase comparison at one size
# ----------------------------------------------------------------------------
def fig_phases(
    results: Dict[str, Any],
    topo: str,
    size: str,
    theme_name: str,
    out_dir: str,
) -> str:
    """Horizontal grouped bars: every phase at a single graph size."""
    theme = THEMES[theme_name]
    style(theme)

    per_impl = results["topologies"][topo][size]
    impls = [i for i in results["impls"] if i in per_impl]

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    bar_h = 0.24
    for row, phase in enumerate(PHASES):
        for k, impl in enumerate(impls):
            stats = per_impl[impl].get(phase)
            if not stats:
                continue
            y = row + (k - (len(impls) - 1) / 2) * (bar_h + 0.02)
            value = stats["median_ms"]
            ax.barh(
                y,
                value,
                height=bar_h,
                color=theme["series"][impl],
                zorder=3,
                label=LABELS[impl] if row == 0 else None,
            )
            ax.annotate(
                f"{value:.2f} ms",
                xy=(value, y),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=7.5,
                color=theme["text_secondary"],
                zorder=4,
            )

    ax.set_yticks(range(len(PHASES)))
    ax.set_yticklabels(
        [PHASE_TITLES[p].split(" (")[0] for p in PHASES],
        color=theme["text"],
    )
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("median ms (log scale) — lower is better")
    ax.get_xaxis().set_major_formatter(FuncFormatter(ms_fmt))
    ax.grid(True, axis="x", linewidth=0.6, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.margins(x=0.30)

    legend = ax.legend(
        frameon=False, loc="lower right", fontsize=8, ncol=1
    )
    for text in legend.get_texts():
        text.set_color(theme["text_secondary"])

    ax.set_title(
        f"{topo} topology · {size} nodes",
        color=theme["text"],
        pad=10,
        loc="left",
    )
    fig.tight_layout()
    return _save(fig, out_dir, f"phases_{topo}_n{size}", theme_name)


# ----------------------------------------------------------------------------
# figure: the total story — cold start vs steady state, and break-even
# ----------------------------------------------------------------------------
def fig_total(
    results: Dict[str, Any],
    topo: str,
    size: str,
    theme_name: str,
    out_dir: str,
) -> str:
    """
    Two panels: what a cold start costs, and how many runs repay it.

    The left panel splits total cold start into import vs first execution — the
    split is the whole point, since import dominates. The right panel plots
    cumulative wall clock against run count so the crossover is readable.
    """
    theme = THEMES[theme_name]
    style(theme)

    per_impl = results["topologies"][topo][size]
    impls = [i for i in results["impls"] if i in per_impl]
    imports = results.get("imports", {})
    import_ms = {
        "swarms-networkx": imports.get(
            "swarms.GraphWorkflow", {}
        ).get("median_ms", 0.0),
        "swarms-rustworkx": imports.get(
            "swarms.GraphWorkflow", {}
        ).get("median_ms", 0.0),
        "langgraph": imports.get("langgraph.StateGraph", {}).get(
            "median_ms", 0.0
        ),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    # --- left: cold start composition -------------------------------------
    xs = range(len(impls))
    for i, impl in enumerate(impls):
        imp = import_ms.get(impl, 0.0)
        first = per_impl[impl]["first_run"]["median_ms"]
        color = theme["series"][impl]
        # Import leg in the series hue; execution leg hatched, separated by a
        # 2px surface gap so the two segments never touch.
        ax1.bar(i, imp, color=color, zorder=3, width=0.5)
        ax1.bar(
            i,
            first,
            bottom=imp * 1.02,
            color=color,
            alpha=0.45,
            hatch="///",
            edgecolor=theme["surface"],
            linewidth=1.4,
            zorder=3,
            width=0.5,
        )
        ax1.annotate(
            f"{imp + first:,.0f} ms",
            xy=(i, imp + first),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=theme["text"],
            zorder=4,
        )
        ax1.annotate(
            f"import {imp:,.0f}",
            xy=(i, imp / 2),
            ha="center",
            va="center",
            fontsize=7,
            color=theme["surface"],
            zorder=4,
        )

    ax1.set_xticks(list(xs))
    ax1.set_xticklabels(
        [LABELS[i].replace(" (", "\n(") for i in impls],
        color=theme["text"],
        fontsize=8,
    )
    ax1.set_ylabel("ms")
    ax1.set_title(
        "Total cold start: import + first execution",
        color=theme["text"],
        pad=10,
        loc="left",
    )
    _finish_axis(ax1, theme)

    # --- right: cumulative cost vs run count -------------------------------
    max_runs = 120
    runs = list(range(0, max_runs + 1))
    crossings = []
    label_entries = []
    for impl in impls:
        imp = import_ms.get(impl, 0.0)
        steady = per_impl[impl]["steady_run"]["median_ms"]
        ys = [imp + steady * r for r in runs]
        color = theme["series"][impl]
        ax2.plot(runs, ys, color=color, linewidth=2.0, zorder=3)
        label_entries.append((runs[-1], ys[-1], LABELS[impl], color))
        crossings.append((impl, imp, steady))

    ax2.set_xlabel("executions")
    ax2.set_ylabel("cumulative ms")
    ax2.set_title(
        "Cumulative wall clock, including import",
        color=theme["text"],
        pad=10,
        loc="left",
    )
    ax2.margins(x=0.22)
    ax2.set_xlim(left=0)  # negative execution counts are meaningless
    _finish_axis(ax2, theme)

    # Mark where the slower-importing framework repays its deficit. The label
    # is pinned near the top of the axes so it never lands on a series line.
    if len(crossings) >= 2:
        fastest = min(crossings, key=lambda c: c[1])
        for impl, imp, steady in crossings:
            if impl == fastest[0]:
                continue
            gain = fastest[2] - steady
            if gain <= 0:
                continue
            n_runs = (imp - fastest[1]) / gain
            if 0 < n_runs <= max_runs:
                ax2.axvline(
                    n_runs,
                    color=theme["muted"],
                    linewidth=1.0,
                    linestyle=(0, (4, 3)),
                    zorder=2,
                )
                ax2.annotate(
                    f"break-even: {n_runs:.0f} runs",
                    xy=(n_runs, 0.965),
                    xycoords=ax2.get_xaxis_transform(),
                    xytext=(6, 0),
                    textcoords="offset points",
                    fontsize=7.5,
                    color=theme["text_secondary"],
                    va="top",
                    zorder=4,
                )
                break

    _direct_labels(ax2, label_entries)

    fig.suptitle(
        f"{topo} topology · {size} nodes — startup cost vs per-run cost",
        color=theme["text"],
        fontsize=12,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, out_dir, f"total_{topo}_n{size}", theme_name)


# ----------------------------------------------------------------------------
# figure: optimization gains (before vs after)
# ----------------------------------------------------------------------------
def fig_gains(
    after: Dict[str, Any],
    before: Dict[str, Any],
    theme_name: str,
    out_dir: str,
) -> str:
    """Speedup factor per phase for the swarms implementations, before → after."""
    theme = THEMES[theme_name]
    style(theme)

    impls = [i for i in after["impls"] if i.startswith("swarms")]
    rows: List[tuple] = []
    for topo, sizes in after["topologies"].items():
        if topo not in before["topologies"]:
            continue
        for size in sorted(sizes, key=int):
            if size not in before["topologies"][topo]:
                continue
            for phase in PHASES:
                for impl in impls:
                    a = (
                        after["topologies"][topo][size]
                        .get(impl, {})
                        .get(phase)
                    )
                    b = (
                        before["topologies"][topo][size]
                        .get(impl, {})
                        .get(phase)
                    )
                    if not a or not b or a["median_ms"] <= 0:
                        continue
                    # Backend goes in the row label too: two impls can produce
                    # the same topo/size/phase row, and color alone must not
                    # be what tells them apart.
                    backend = impl.split("-", 1)[-1]
                    rows.append(
                        (
                            f"{topo} n={size} · {phase} · {backend}",
                            impl,
                            b["median_ms"] / a["median_ms"],
                        )
                    )

    rows = [r for r in rows if r[2] >= 1.15]
    rows.sort(key=lambda r: r[2])
    rows = rows[-22:]
    if not rows:
        return ""

    fig, ax = plt.subplots(figsize=(8.6, max(4.0, 0.30 * len(rows))))
    for i, (label, impl, speedup) in enumerate(rows):
        color = theme["series"][impl]
        ax.barh(i, speedup, height=0.62, color=color, zorder=3)
        ax.annotate(
            f"{speedup:.1f}x",
            xy=(speedup, i),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
            color=theme["text_secondary"],
            zorder=4,
        )

    ax.axvline(1.0, color=theme["axis"], linewidth=1.0, zorder=2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        [r[0] for r in rows], fontsize=7.5, color=theme["text"]
    )
    ax.set_xscale("log")
    ax.set_xlabel("speedup factor (before ÷ after, log scale)")
    ax.get_xaxis().set_major_formatter(
        FuncFormatter(lambda v, p: f"{v:g}x")
    )
    ax.grid(True, axis="x", linewidth=0.6, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.margins(x=0.18)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=theme["series"][i])
        for i in impls
    ]
    legend = ax.legend(
        handles,
        [LABELS[i] for i in impls],
        frameon=False,
        loc="lower right",
        fontsize=8,
    )
    for text in legend.get_texts():
        text.set_color(theme["text_secondary"])

    ax.set_title(
        "Optimization gains — swarms GraphWorkflow, before vs after",
        color=theme["text"],
        pad=10,
        loc="left",
    )
    fig.tight_layout()
    return _save(fig, out_dir, "optimization_gains", theme_name)


def _save(fig, out_dir: str, name: str, theme_name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.{theme_name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "results", help="results JSON from graph_workflow_bench.py"
    )
    ap.add_argument(
        "--baseline",
        default=None,
        help="optional earlier results JSON to compute before/after gains",
    )
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--themes", default="light,dark", help="comma separated"
    )
    args = ap.parse_args()

    with open(args.results) as fh:
        after = json.load(fh)
    before = None
    if args.baseline:
        with open(args.baseline) as fh:
            before = json.load(fh)

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.results)), "figures"
    )

    written: List[str] = []
    for theme_name in args.themes.split(","):
        theme_name = theme_name.strip()
        for topo in after["topologies"]:
            written.append(
                fig_scaling(after, topo, theme_name, out_dir)
            )
            largest = max(after["topologies"][topo], key=int)
            written.append(
                fig_phases(after, topo, largest, theme_name, out_dir)
            )
            if after.get("imports"):
                written.append(
                    fig_total(
                        after, topo, largest, theme_name, out_dir
                    )
                )
        if before:
            path = fig_gains(after, before, theme_name, out_dir)
            if path:
                written.append(path)

    for path in written:
        print(path)
    print(f"\n{len(written)} figures → {out_dir}")


if __name__ == "__main__":
    main()
