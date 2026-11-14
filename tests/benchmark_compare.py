"""
Compare Swarms benchmark numbers between two checkouts and fail on a regression.

Runs tests/benchmark.py and tests/benchmark_agent_rearrange.py from this
checkout against the swarms package of each checkout, alternating between
the two for several rounds, then compares the median of every metric.
A gated metric regresses when the head median grows past both its
relative and its absolute limit, or when any run lacks it. Metrics that
vary too much between runs to gate are reported without a limit.

Run from the repo root:
    python tests/benchmark_compare.py --base ../swarms-master --head .
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

SCRIPTS = Path(__file__).resolve().parent

FANOUT_ARGS = ["--frameworks", "swarms"]
REARRANGE_ARGS = [
    "--frameworks",
    "swarms",
    "--sizes",
    "128",
    "--topologies",
    "chain,fanout",
]
REARRANGE_FIELDS = (
    "overhead_ms",
    "build_ms",
    "rearrange_ms",
    "built_kb_per_agent",
    "peak_mb",
    "peak_threads",
    "model_calls",
)


@dataclass(frozen=True)
class Limit:
    """How far a metric may grow before it counts as a regression.

    Args:
        relative: Allowed growth as a fraction of the base median.
        absolute: Allowed growth in the metric's own unit.
        exact: True when any change at all is a regression.
    """

    relative: float = 0.0
    absolute: float = 0.0
    exact: bool = False


LIMITS: Dict[str, Optional[Limit]] = {
    "fanout: cold import (s)": Limit(0.2, 0.05),
    "fanout: construction per agent, workflow included (ms)": Limit(
        0.3, 0.005
    ),
    "fanout: memory per agent, workflow included (KB)": Limit(
        0.05, 0.2
    ),
    "fanout: 100-agent fan-out wall (s)": Limit(0.05, 0.01),
    "fanout: fan-out wall minus model delay (s)": None,
    "fanout: fan-out peak memory, agents included (MB)": Limit(
        0.05, 0.1
    ),
    "fanout: fan-out peak threads": Limit(0.0, 2),
    "fanout: model calls per fan-out": Limit(exact=True),
    "chain 128: overhead_ms": Limit(0.25, 10),
    "chain 128: build_ms": None,
    "chain 128: rearrange_ms": None,
    "chain 128: built_kb_per_agent": Limit(0.05, 0.2),
    "chain 128: peak_mb": Limit(0.05, 0.2),
    "chain 128: peak_threads": Limit(0.0, 2),
    "chain 128: model_calls": Limit(exact=True),
    "fanout 128: overhead_ms": Limit(0.5, 10),
    "fanout 128: build_ms": None,
    "fanout 128: rearrange_ms": None,
    "fanout 128: built_kb_per_agent": Limit(0.05, 0.2),
    "fanout 128: peak_mb": None,
    "fanout 128: peak_threads": None,
    "fanout 128: model_calls": Limit(exact=True),
}

FAILING = ("regressed", "missing")


@dataclass(frozen=True)
class Row:
    """One metric compared between base and head.

    Args:
        metric: The metric name.
        base: Median over the base runs, or None when base lacks it.
        head: Median over the head runs, or None when head lacks it.
        noise: The larger relative spread of the two sides' runs.
        limit: The limit applied, or None for a report-only metric.
        status: One of ok, regressed, missing, new or report only.
    """

    metric: str
    base: Optional[float]
    head: Optional[float]
    noise: Optional[float]
    limit: Optional[Limit]
    status: str


def parse_results(fanout: dict, rearrange: list) -> Dict[str, float]:
    """Turn the two benchmarks' JSON output into named metrics.

    Args:
        fanout: What tests/benchmark.py wrote.
        rearrange: What tests/benchmark_agent_rearrange.py wrote.

    Returns:
        Every metric value, keyed by metric name.

    Raises:
        ValueError: If a cell reports an error or a value is not a
            finite number.
    """
    metrics: Dict[str, float] = {}

    for name, value in fanout.get("swarms", {}).items():
        metrics[f"fanout: {name}"] = value

    for cell in rearrange:
        if "error" in cell:
            raise ValueError(f"AgentRearrange: {cell['error']}")
        prefix = f"{cell['topology']} {cell['agents']}"
        for field in REARRANGE_FIELDS:
            if cell.get(field) is not None:
                metrics[f"{prefix}: {field}"] = cell[field]

    for name, value in metrics.items():
        if isinstance(value, bool) or not isinstance(
            value, (int, float)
        ):
            raise ValueError(f"{name} is not a number: {value!r}")
        if not math.isfinite(value):
            raise ValueError(f"{name} is not finite: {value!r}")

    return {name: float(value) for name, value in metrics.items()}


def run_benchmarks(tree: Path, work: Path) -> Dict[str, float]:
    """Run both benchmarks against the swarms package in one checkout.

    Args:
        tree: Checkout whose swarms package is measured.
        work: Folder for the JSON the benchmarks write.

    Returns:
        Every metric value, keyed by metric name.

    Raises:
        RuntimeError: If a benchmark fails or reports a bad value.
    """
    env = {
        **os.environ,
        "PYTHONPATH": str(tree),
        "SWARMS_TELEMETRY_ON": "False",
    }
    fanout_out = work / "fanout.json"
    rearrange_out = work / "rearrange.json"

    for script, args, out in (
        ("benchmark.py", FANOUT_ARGS, fanout_out),
        (
            "benchmark_agent_rearrange.py",
            REARRANGE_ARGS,
            rearrange_out,
        ),
    ):
        command = [
            sys.executable,
            str(SCRIPTS / script),
            *args,
            "--out",
            str(out),
        ]
        done = subprocess.run(
            command, cwd=tree, env=env, capture_output=True, text=True
        )
        if done.returncode != 0 or not out.exists():
            raise RuntimeError(
                f"{script} failed on {tree}:\n{done.stderr[-2000:]}"
            )

    try:
        return parse_results(
            json.loads(fanout_out.read_text()),
            json.loads(rearrange_out.read_text()),
        )
    except ValueError as error:
        raise RuntimeError(f"{tree}: {error}") from error


def spread(values: List[float]) -> Optional[float]:
    """Measure how far apart repeated runs of one metric landed.

    Args:
        values: The metric's value in each run.

    Returns:
        The range divided by the median, or None when it is undefined.
    """
    if len(values) < 2 or not statistics.median(values):
        return None
    return (max(values) - min(values)) / abs(
        statistics.median(values)
    )


def compare(
    base_runs: List[Dict[str, float]],
    head_runs: List[Dict[str, float]],
    limits: Dict[str, Optional[Limit]] = LIMITS,
) -> List[Row]:
    """Compare the median of every metric between base and head.

    A gated metric must appear in every run on both sides, so a renamed
    or dropped measurement fails instead of losing its gate.

    Args:
        base_runs: Metrics from each base run.
        head_runs: Metrics from each head run.
        limits: The limit for each metric; None marks it report only.

    Returns:
        One row per metric, in a stable order.
    """
    gated = {name for name, limit in limits.items() if limit}
    names = sorted(
        gated
        | {name for run in base_runs + head_runs for name in run}
    )
    rows: List[Row] = []

    for name in names:
        base_values = [run[name] for run in base_runs if name in run]
        head_values = [run[name] for run in head_runs if name in run]
        base = statistics.median(base_values) if base_values else None
        head = statistics.median(head_values) if head_values else None
        noises = [
            value
            for value in (spread(base_values), spread(head_values))
            if value is not None
        ]
        limit = limits.get(name)
        complete = (
            bool(base_runs)
            and bool(head_runs)
            and len(base_values) == len(base_runs)
            and len(head_values) == len(head_runs)
        )

        if limit and not complete:
            status = "missing"
        elif head is None:
            status = "report only"
        elif base is None:
            status = "new"
        elif limit is None:
            status = "report only"
        elif limit.exact:
            status = "regressed" if head != base else "ok"
        else:
            grew_past_limit = (
                head > base * (1 + limit.relative)
                and head - base > limit.absolute
            )
            status = "regressed" if grew_past_limit else "ok"

        rows.append(
            Row(
                name,
                base,
                head,
                max(noises) if noises else None,
                limit,
                status,
            )
        )

    return rows


def describe(limit: Optional[Limit]) -> str:
    """Describe a limit for the results table.

    Args:
        limit: The limit, or None for a report-only metric.

    Returns:
        A short human-readable description.
    """
    if limit is None:
        return "report only"
    if limit.exact:
        return "must not change"
    if limit.relative == 0:
        return f"+{limit.absolute:g}"
    return f"+{limit.relative:.0%} and +{limit.absolute:g}"


def render(rows: List[Row], rounds: int) -> str:
    """Render the comparison as a Markdown table.

    Args:
        rows: The compared metrics.
        rounds: How many runs each side had.

    Returns:
        The Markdown text.
    """
    failed = [row for row in rows if row.status in FAILING]
    gated = sum(1 for row in rows if row.limit)
    verdict = (
        f"**{len(failed)} regression(s)**"
        if failed
        else "No regressions"
    )
    lines = [
        "## Swarms benchmark: base vs head",
        "",
        f"{verdict}. {gated} metrics are gated and "
        f"{len(rows) - gated} are report only. Each side ran "
        f"{rounds} times, alternating on the same machine. A gated "
        "metric regresses when its median grows past both of its "
        "limits, or when a run lacks it. Noise is the larger spread "
        "of either side's runs.",
        "",
        "| metric | base | head | change | noise | limit | status |",
        "|---|---|---|---|---|---|---|",
    ]

    for row in rows:
        change = ""
        if row.base and row.head is not None:
            change = f"{(row.head - row.base) / row.base:+.1%}"
        base = "" if row.base is None else f"{row.base:.3f}"
        head = "" if row.head is None else f"{row.head:.3f}"
        noise = "" if row.noise is None else f"{row.noise:.0%}"
        lines.append(
            f"| {row.metric} | {base} | {head} | {change} | {noise} "
            f"| {describe(row.limit)} | {row.status} |"
        )

    return "\n".join(lines) + "\n"


def positive_int(text: str) -> int:
    """Parse a whole number of at least 1 for the command line.

    Args:
        text: The value given on the command line.

    Returns:
        The parsed number.

    Raises:
        argparse.ArgumentTypeError: If it is not at least 1.
    """
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main() -> None:
    """Run the comparison and exit with 1 when a gated metric fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--head", required=True, type=Path)
    parser.add_argument("--rounds", type=positive_int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    base_runs: List[Dict[str, float]] = []
    head_runs: List[Dict[str, float]] = []

    with tempfile.TemporaryDirectory() as folder:
        for round_number in range(args.rounds):
            sides = [
                (args.base, base_runs),
                (args.head, head_runs),
            ]
            if round_number % 2:
                sides.reverse()
            for tree, runs in sides:
                work = Path(folder) / f"{len(base_runs + head_runs)}"
                work.mkdir()
                runs.append(run_benchmarks(tree.resolve(), work))
                print(
                    f"round {round_number + 1}/{args.rounds}: "
                    f"measured {tree}",
                    file=sys.stderr,
                    flush=True,
                )

    rows = compare(base_runs, head_runs)
    report = render(rows, args.rounds)
    print(report)

    if args.out:
        args.out.write_text(
            json.dumps(
                {"base": base_runs, "head": head_runs}, indent=1
            )
        )

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as handle:
            handle.write(report)

    if any(row.status in FAILING for row in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
