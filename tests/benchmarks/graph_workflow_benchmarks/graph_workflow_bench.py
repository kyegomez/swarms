"""
GraphWorkflow performance benchmark — swarms vs LangGraph.

Measures the framework overhead that is *not* LLM latency:

  1. import      — cold interpreter, `from X import Y` (subprocess, baselined)
  2. build       — add N nodes + M edges to an uncompiled graph
  3. compile     — topological pre-computation / validation
  4. first_run   — cold: build + compile + one execution
  5. steady_run  — repeated execution of an already-compiled graph

Nodes are no-ops, so every measured microsecond is orchestration overhead.
Derived metrics (total cold start, break-even run count) are computed from
those five and written alongside them.

Usage
-----
    python3 tests/benchmarks/graph_workflow_benchmarks/graph_workflow_bench.py
    python3 tests/benchmarks/graph_workflow_benchmarks/graph_workflow_bench.py --sizes 10,50,200 --repeats 9
    python3 tests/benchmarks/graph_workflow_benchmarks/graph_workflow_bench.py --out tests/benchmarks/graph_workflow_benchmarks/results/after.json

Then visualise:

    python3 tests/benchmarks/graph_workflow_benchmarks/plot_results.py tests/benchmarks/graph_workflow_benchmarks/results/after.json \\
        --baseline tests/benchmarks/graph_workflow_benchmarks/results/before.json

Fairness notes
--------------
* Both frameworks execute the identical topology, node-for-node and edge-for-edge.
* LangGraph's default 25-superstep cap is raised so deep graphs are not truncated.
* LangGraph state accumulation is measured under BOTH reducers (see ``--lg-reducer``):
  ``counter`` keeps state O(1) per superstep, ``list`` accumulates every node's
  output the way swarms' ``prev_outputs``/conversation inherently does. Reporting
  both isolates reducer cost from orchestration cost.
"""

import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

os.environ.setdefault("SWARMS_TELEMETRY_ON", "false")

def _find_repo_root() -> str:
    """
    Walk up from this file until the directory containing the `swarms` package.

    Always measure the working tree, never a `pip install`ed copy: running this
    script puts its own directory on `sys.path`, not the repo root, so an
    installed swarms would silently win and the numbers would describe the
    wrong code. Searching for the package rather than counting parent
    directories keeps this correct if the benchmark is ever moved again.

    Returns:
        str: Absolute path to the repo root, or this file's directory if no
        `swarms/` package was found above it.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    current = here
    while True:
        if os.path.isfile(
            os.path.join(current, "swarms", "__init__.py")
        ):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return here
        current = parent


_REPO_ROOT = _find_repo_root()
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in [_REPO_ROOT, os.environ.get("PYTHONPATH", "")] if p
)

# Results live beside this script, not at the repo root.
_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results"
)

PHASES = ["build", "compile", "first_run", "steady_run"]


# ----------------------------------------------------------------------------
# environment capture — a benchmark without provenance is an anecdote
# ----------------------------------------------------------------------------
def environment() -> Dict[str, Any]:
    """Record the machine and library versions the numbers were produced on."""

    def version(module_name: str) -> str:
        try:
            import importlib.metadata as md

            return md.version(module_name)
        except Exception:
            return "not-installed"

    try:
        git_rev = (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                cwd=_REPO_ROOT,
            ).stdout.strip()
            or "unknown"
        )
        git_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "swarms/"],
                capture_output=True,
                text=True,
                cwd=_REPO_ROOT,
            ).stdout.strip()
        )
    except Exception:
        git_rev, git_dirty = "unknown", False

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "git_rev": git_rev,
        "git_dirty_swarms": git_dirty,
        "versions": {
            name: version(name)
            for name in (
                "swarms",
                "langgraph",
                "networkx",
                "rustworkx",
            )
        },
    }


# ----------------------------------------------------------------------------
# timing
# ----------------------------------------------------------------------------
def bench(
    fn: Callable[[], object],
    repeats: int,
    warmup: int,
    setup: Callable[[], None] = None,
) -> Dict[str, Any]:
    """
    Time ``fn`` and summarise the distribution.

    Garbage collection is disabled inside each timed region and forced between
    samples, so a collection triggered by an earlier sample is never billed to a
    later one.

    Args:
        fn (Callable[[], object]): The zero-argument callable to time.
        repeats (int): Number of timed samples to collect.
        warmup (int): Untimed iterations run first and discarded.
        setup (Callable[[], None]): Optional per-sample setup, run untimed.

    Returns:
        Dict[str, Any]: ``min_ms``/``median_ms``/``mean_ms``/``p95_ms``/
        ``stdev_ms``/``n`` plus the raw ``samples_ms`` list.
    """
    for _ in range(warmup):
        if setup:
            setup()
        fn()

    samples: List[float] = []
    for _ in range(repeats):
        if setup:
            setup()
        gc.collect()
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            start = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - start) * 1000.0)
        finally:
            if gc_was_enabled:
                gc.enable()

    ordered = sorted(samples)
    p95_idx = min(
        len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))
    )
    return {
        "min_ms": ordered[0],
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "p95_ms": ordered[p95_idx],
        "stdev_ms": (
            statistics.stdev(samples) if len(samples) > 1 else 0.0
        ),
        "n": len(samples),
        "samples_ms": samples,
    }


def cold_import_ms(statement: str, repeats: int) -> Dict[str, Any]:
    """
    Median cost of ``statement`` in a fresh interpreter, minus interpreter start.

    Args:
        statement (str): Import statement to time.
        repeats (int): Subprocess launches to median over.

    Returns:
        Dict[str, Any]: ``median_ms`` (baseline-subtracted) and raw samples.
    """
    baseline = _subprocess_samples("pass", repeats)
    loaded = _subprocess_samples(statement, repeats)
    base_median = statistics.median(baseline)
    return {
        "median_ms": max(
            0.0, statistics.median(loaded) - base_median
        ),
        "raw_median_ms": statistics.median(loaded),
        "interpreter_baseline_ms": base_median,
        "samples_ms": loaded,
        "n": len(loaded),
    }


def _subprocess_samples(statement: str, repeats: int) -> List[float]:
    code = (
        "import time;_t=time.perf_counter()\n"
        f"{statement}\n"
        "print((time.perf_counter()-_t)*1000)"
    )
    env = dict(os.environ, SWARMS_TELEMETRY_ON="false")
    samples = []
    for _ in range(repeats):
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
        )
        if out.returncode != 0:
            raise RuntimeError(
                f"import probe failed for {statement!r}:\n{out.stderr}"
            )
        samples.append(float(out.stdout.strip().splitlines()[-1]))
    return samples


# ----------------------------------------------------------------------------
# topologies — identical shape for both frameworks
# ----------------------------------------------------------------------------
def chain_edges(n: int) -> List[Tuple[str, str]]:
    """Pure depth: n0 -> n1 -> ... -> n(n-1). One node per layer."""
    return [(f"n{i}", f"n{i + 1}") for i in range(n - 1)]


def wide_edges(n: int) -> List[Tuple[str, str]]:
    """Pure width: one root fanning out to n-1 leaves. Two layers."""
    return [("n0", f"n{i}") for i in range(1, n)]


def diamond_edges(n: int) -> List[Tuple[str, str]]:
    """Fan-out then fan-in: n0 -> n1..n(n-2) -> n(n-1). Three layers."""
    if n < 3:
        return chain_edges(n)
    edges = [("n0", f"n{i}") for i in range(1, n - 1)]
    edges += [(f"n{i}", f"n{n - 1}") for i in range(1, n - 1)]
    return edges


def layered_edges(n: int, width: int = 4) -> List[Tuple[str, str]]:
    """Stacked layers of ``width`` nodes, fully connected between layers."""
    edges = []
    layers = [
        [f"n{i}" for i in range(s, min(s + width, n))]
        for s in range(0, n, width)
    ]
    for a, b in zip(layers, layers[1:]):
        for src in a:
            for dst in b:
                edges.append((src, dst))
    return edges


def tree_edges(n: int) -> List[Tuple[str, str]]:
    """Binary tree: node i has children 2i+1 and 2i+2. Log-depth fan-out."""
    edges = []
    for i in range(n):
        for child in (2 * i + 1, 2 * i + 2):
            if child < n:
                edges.append((f"n{i}", f"n{child}"))
    return edges


TOPOLOGIES = {
    "chain": chain_edges,
    "wide": wide_edges,
    "diamond": diamond_edges,
    "layered": layered_edges,
    "tree": tree_edges,
}


# ----------------------------------------------------------------------------
# swarms harness
# ----------------------------------------------------------------------------
class NoOpAgent:
    """Minimal stand-in for `Agent` — isolates graph overhead from LLM latency."""

    __slots__ = ("agent_name",)

    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    def run(self, task=None, img=None, *args, **kwargs):
        return "ok"


def swarms_build(n: int, edges, backend: str, auto_compile: bool):
    """Construct a GraphWorkflow with ``n`` no-op agents and the given edges."""
    from swarms.structs.graph_workflow import GraphWorkflow

    wf = GraphWorkflow(auto_compile=auto_compile, backend=backend)
    for i in range(n):
        wf.add_node(NoOpAgent(f"n{i}"))
    for src, dst in edges:
        wf.add_edge(src, dst)
    return wf


def swarms_suite(
    n: int, edges, backend: str, repeats: int, warmup: int
) -> Dict[str, Any]:
    """Run every phase against swarms' GraphWorkflow."""
    res: Dict[str, Any] = {}

    res["build"] = bench(
        lambda: swarms_build(n, edges, backend, auto_compile=False),
        repeats,
        warmup,
    )

    # compile() is idempotent, so each sample needs a fresh uncompiled graph;
    # building it is done in untimed setup.
    holder: Dict[str, Any] = {}

    res["compile"] = bench(
        lambda: holder["wf"].compile(),
        repeats,
        warmup,
        setup=lambda: holder.__setitem__(
            "wf", swarms_build(n, edges, backend, auto_compile=False)
        ),
    )

    def _cold():
        wf = swarms_build(n, edges, backend, auto_compile=True)
        wf.run("benchmark task")

    res["first_run"] = bench(_cold, repeats, warmup)

    warm = swarms_build(n, edges, backend, auto_compile=True)
    warm.run("warmup")
    res["steady_run"] = bench(
        lambda: warm.run("benchmark task"), repeats, warmup
    )
    return res


# ----------------------------------------------------------------------------
# langgraph harness
# ----------------------------------------------------------------------------
def _lg_state(reducer: str):
    """
    Build the state schema.

    Args:
        reducer (str): ``"counter"`` keeps accumulated state O(1) per superstep;
            ``"list"`` accumulates every node's output, matching what swarms'
            ``prev_outputs`` and conversation do inherently. Measuring both
            separates reducer cost from orchestration cost.

    Returns:
        type: A TypedDict subclass usable as a StateGraph schema.
    """
    import operator
    from typing import Annotated, List

    from typing_extensions import TypedDict

    if reducer == "counter":

        class State(TypedDict):
            task: str
            out: Annotated[int, operator.add]

    else:

        class State(TypedDict):
            task: str
            out: Annotated[List[str], operator.add]

    return State


def langgraph_build(n: int, edges, compile_it: bool, reducer: str):
    """Construct the equivalent LangGraph StateGraph with no-op nodes."""
    from langgraph.graph import END, START, StateGraph

    if reducer == "counter":

        def make(_name):
            def node(state):
                return {"out": 1}

            return node

    else:

        def make(name):
            def node(state):
                return {"out": [f"{name}:ok"]}

            return node

    g = StateGraph(_lg_state(reducer))
    for i in range(n):
        g.add_node(f"n{i}", make(f"n{i}"))

    targets = {dst for _, dst in edges}
    sources = {src for src, _ in edges}
    for i in range(n):
        name = f"n{i}"
        if name not in targets:
            g.add_edge(START, name)
        if name not in sources:
            g.add_edge(name, END)
    for src, dst in edges:
        g.add_edge(src, dst)

    return g.compile() if compile_it else g


def langgraph_suite(
    n: int, edges, repeats: int, warmup: int, reducer: str
) -> Dict[str, Any]:
    """Run every phase against LangGraph."""
    res: Dict[str, Any] = {}

    res["build"] = bench(
        lambda: langgraph_build(n, edges, False, reducer),
        repeats,
        warmup,
    )

    holder: Dict[str, Any] = {}
    res["compile"] = bench(
        lambda: holder["g"].compile(),
        repeats,
        warmup,
        setup=lambda: holder.__setitem__(
            "g", langgraph_build(n, edges, False, reducer)
        ),
    )

    seed = {
        "task": "benchmark task",
        "out": 0 if reducer == "counter" else [],
    }
    # Deep graphs exceed LangGraph's default 25-superstep cap; raise it so both
    # frameworks execute the identical topology rather than a truncated one.
    cfg = {"recursion_limit": 100_000}

    def _cold():
        app = langgraph_build(n, edges, True, reducer)
        app.invoke(dict(seed), config=cfg)

    res["first_run"] = bench(_cold, repeats, warmup)

    warm = langgraph_build(n, edges, True, reducer)
    warm.invoke(dict(seed), config=cfg)
    res["steady_run"] = bench(
        lambda: warm.invoke(dict(seed), config=cfg), repeats, warmup
    )
    return res


# ----------------------------------------------------------------------------
# derived metrics
# ----------------------------------------------------------------------------
def derive(results: Dict[str, Any]) -> None:
    """
    Add total-cold-start and break-even figures in place.

    ``total_cold_start_ms`` is import + first_run: what a one-shot script pays.
    ``breakeven_runs`` is how many executions it takes for a slower-to-import
    framework to repay its startup deficit against the fastest importer.
    """
    imports = results.get("imports", {})
    if not imports:
        return

    import_for = {
        "swarms-networkx": imports.get(
            "swarms.GraphWorkflow", {}
        ).get("median_ms"),
        "swarms-rustworkx": imports.get(
            "swarms.GraphWorkflow", {}
        ).get("median_ms"),
        "langgraph": imports.get("langgraph.StateGraph", {}).get(
            "median_ms"
        ),
    }

    for topo, sizes in results["topologies"].items():
        for size, per_impl in sizes.items():
            cold = {}
            for impl, phases in per_impl.items():
                imp = import_for.get(impl)
                if imp is None:
                    continue
                cold[impl] = imp + phases["first_run"]["median_ms"]
                phases["total_cold_start_ms"] = cold[impl]

            if not cold:
                continue

            fastest_cold = min(cold, key=cold.get)
            for impl, phases in per_impl.items():
                if impl not in cold:
                    continue
                deficit = cold[impl] - cold[fastest_cold]
                gain = (
                    per_impl[fastest_cold]["steady_run"]["median_ms"]
                    - phases["steady_run"]["median_ms"]
                )
                phases["breakeven_runs"] = (
                    None
                    if gain <= 0
                    else (0 if deficit <= 0 else deficit / gain)
                )
                phases["breakeven_vs"] = fastest_cold


# ----------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------
def report(results: Dict[str, Any]) -> str:
    """Render the results as a Markdown report (also printed to stdout)."""
    env = results["env"]
    lines = [
        "# GraphWorkflow benchmark — swarms vs LangGraph",
        "",
        f"*{env['timestamp']} · {env['platform']} · "
        f"{env['cpu_count']} cores · Python {env['python']} · "
        f"swarms @ {env['git_rev']}"
        f"{' (dirty)' if env['git_dirty_swarms'] else ''}*",
        "",
        "Versions: "
        + ", ".join(f"{k} {v}" for k, v in env["versions"].items()),
        "",
        f"Timing: median of {results['config']['repeats']} samples after "
        f"{results['config']['warmup']} warmup iterations, GC disabled inside "
        f"each timed region. LangGraph reducer: "
        f"`{results['config']['lg_reducer']}`.",
        "",
        "All nodes are no-ops — every number is orchestration overhead, not "
        "LLM latency.",
        "",
    ]

    if results.get("imports"):
        lines += [
            "## Import (cold interpreter, baseline-subtracted)",
            "",
            "| target | median ms |",
            "| --- | ---: |",
        ]
        for name, stats in results["imports"].items():
            lines.append(f"| `{name}` | {stats['median_ms']:.1f} |")
        lines.append("")

    impls = results["impls"]
    for topo, sizes in results["topologies"].items():
        lines += [f"## Topology: {topo}", ""]
        header = "| nodes | phase | " + " | ".join(impls) + " |"
        lines.append(header)
        lines.append(
            "| ---: | --- | "
            + " | ".join("---:" for _ in impls)
            + " |"
        )
        for size in sorted(sizes, key=int):
            per_impl = sizes[size]
            for phase in PHASES:
                cells = []
                for impl in impls:
                    stats = per_impl.get(impl, {}).get(phase)
                    cells.append(
                        f"{stats['median_ms']:.3f}" if stats else "—"
                    )
                lines.append(
                    f"| {size} | {phase} | "
                    + " | ".join(cells)
                    + " |"
                )
        lines.append("")

    md = "\n".join(lines)
    print(md)
    return md


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--sizes", default="10,50,200")
    ap.add_argument("--repeats", type=int, default=9)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--import-repeats", type=int, default=5)
    ap.add_argument(
        "--topologies", default="chain,wide,diamond,layered,tree"
    )
    ap.add_argument(
        "--impls",
        default="swarms-networkx,swarms-rustworkx,langgraph",
    )
    ap.add_argument(
        "--lg-reducer",
        default="list",
        choices=["counter", "list"],
        help=(
            "LangGraph state reducer. 'list' accumulates every node's output "
            "(comparable to swarms' prev_outputs); 'counter' keeps state O(1) "
            "to isolate reducer cost from orchestration cost."
        ),
    )
    ap.add_argument("--skip-imports", action="store_true")
    ap.add_argument(
        "--out",
        default=os.path.join(
            _RESULTS_DIR, "latest.json"
        ),
        help="where to write raw results JSON (a .md report goes beside it)",
    )
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    impls = [s.strip() for s in args.impls.split(",")]
    topologies = [t.strip() for t in args.topologies.split(",")]

    # Import both frameworks up front, then silence their loggers, so neither
    # one-off import cost nor log I/O leaks into the phase timings.
    if any(i.startswith("swarms") for i in impls):
        import swarms.structs.graph_workflow as _gw

        print(f"swarms module: {_gw.__file__}", file=sys.stderr)
    if "langgraph" in impls:
        import langgraph.graph  # noqa: F401

    import logging

    logging.disable(logging.CRITICAL)
    try:
        from loguru import logger as _loguru

        _loguru.remove()
    except Exception:
        pass

    results: Dict[str, Any] = {
        "env": environment(),
        "config": {
            "sizes": sizes,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "topologies": topologies,
            "lg_reducer": args.lg_reducer,
        },
        "impls": impls,
        "imports": {},
        "topologies": {},
    }

    if not args.skip_imports:
        print("Measuring cold import times...", file=sys.stderr)
        probes = {}
        if any(i.startswith("swarms") for i in impls):
            probes["swarms (top-level)"] = "import swarms"
            probes["swarms.GraphWorkflow"] = (
                "from swarms.structs.graph_workflow import GraphWorkflow"
            )
        if "langgraph" in impls:
            probes["langgraph.StateGraph"] = (
                "from langgraph.graph import StateGraph"
            )
        for name, stmt in probes.items():
            results["imports"][name] = cold_import_ms(
                stmt, args.import_repeats
            )

    for topo in topologies:
        edge_fn = TOPOLOGIES[topo]
        results["topologies"][topo] = {}
        for n in sizes:
            edges = edge_fn(n)
            print(
                f"  {topo} n={n} ({len(edges)} edges)...",
                file=sys.stderr,
            )
            per_impl = {}
            if "swarms-networkx" in impls:
                per_impl["swarms-networkx"] = swarms_suite(
                    n, edges, "networkx", args.repeats, args.warmup
                )
            if "swarms-rustworkx" in impls:
                per_impl["swarms-rustworkx"] = swarms_suite(
                    n, edges, "rustworkx", args.repeats, args.warmup
                )
            if "langgraph" in impls:
                per_impl["langgraph"] = langgraph_suite(
                    n,
                    edges,
                    args.repeats,
                    args.warmup,
                    args.lg_reducer,
                )
            results["topologies"][topo][str(n)] = per_impl

    derive(results)
    md = report(results)

    os.makedirs(
        os.path.dirname(os.path.abspath(args.out)), exist_ok=True
    )
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    md_path = os.path.splitext(args.out)[0] + ".md"
    with open(md_path, "w") as fh:
        fh.write(md)
    print(f"\nRaw results → {args.out}", file=sys.stderr)
    print(f"Report      → {md_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
