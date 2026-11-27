import math

import pytest

from benchmark_compare import Limit, compare, parse_results

LIMITS = {
    "memory (KB)": Limit(0.1, 0.5),
    "threads": Limit(0.0, 2),
    "model calls": Limit(exact=True),
    "wall (s)": None,
}


def _runs(values):
    return [{} if value is None else {"m": value} for value in values]


@pytest.mark.parametrize(
    "metric, base, head, status",
    [
        ("memory (KB)", [6.5, 6.6, 6.5], [6.6, 6.6, 6.7], "ok"),
        (
            "memory (KB)",
            [6.5, 6.6, 6.5],
            [7.9, 8.0, 8.1],
            "regressed",
        ),
        ("memory (KB)", [1.0, 1.0, 1.0], [1.3, 1.3, 1.3], "ok"),
        ("memory (KB)", [6.5, 6.6, 6.5], [3.0, 3.1, 3.0], "ok"),
        ("threads", [2, 2, 2], [4, 4, 4], "ok"),
        ("threads", [2, 2, 2], [101, 101, 101], "regressed"),
        ("model calls", [100, 100, 100], [99, 99, 99], "regressed"),
        ("wall (s)", [0.2, 0.2, 0.2], [0.9, 0.9, 0.9], "report only"),
        (
            "memory (KB)",
            [6.5, 6.6, 6.5],
            [6.5, None, None],
            "missing",
        ),
        ("memory (KB)", [None, None], [6.5, 6.6], "missing"),
        ("memory (KB)", [None, None], [None, None], "missing"),
        ("memory (KB)", [], [], "missing"),
        ("wall (s)", [0.2, 0.2], [None, None], "report only"),
    ],
    ids=[
        "within-noise",
        "grew-past-both-limits",
        "relative-only",
        "improved",
        "within-absolute",
        "thread-per-agent",
        "exact-changed",
        "report-only",
        "missing-from-some-head-runs",
        "missing-from-base",
        "renamed-away-on-both-sides",
        "no-runs",
        "report-only-dropped",
    ],
)
def test_compare_gates_growth_and_requires_every_run(
    metric, base, head, status
):
    (row,) = compare(_runs(base), _runs(head), {"m": LIMITS[metric]})

    assert row.status == status


@pytest.mark.parametrize(
    "fanout, rearrange",
    [
        ({"swarms": {"wall (s)": math.nan}}, []),
        ({"swarms": {"wall (s)": math.inf}}, []),
        ({"swarms": {"wall (s)": "not installed"}}, []),
        ({"swarms": {"wall (s)": True}}, []),
        (
            {"swarms": {}},
            [{"topology": "chain", "agents": 128, "error": "boom"}],
        ),
        (
            {"swarms": {}},
            [
                {
                    "topology": "chain",
                    "agents": 128,
                    "peak_mb": -math.inf,
                }
            ],
        ),
    ],
    ids=["nan", "inf", "text", "bool", "error-cell", "negative-inf"],
)
def test_parse_results_rejects_values_that_cannot_be_compared(
    fanout, rearrange
):
    with pytest.raises(ValueError):
        parse_results(fanout, rearrange)
