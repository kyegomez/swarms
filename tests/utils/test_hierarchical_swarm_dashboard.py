"""Tests for HierarchicalSwarmDashboard UI fixes.

Covers the bug fixed in PR #1917 / issue #1914:
  director orders loop was missing [:5] cap — all orders rendered and
  the "...and N more" truncation label was factually wrong.

Uses importlib to load the dashboard module directly so the full
swarms package (and its heavy dependencies) are not triggered.
"""

import importlib.util
import io
from pathlib import Path

import pytest
from rich.console import Console

# Load only the dashboard module — avoids triggering swarms/__init__.py
_DASHBOARD_PATH = (
    Path(__file__).parent.parent.parent
    / "swarms" / "utils" / "hierarchical_swarm_dashboard.py"
)
_spec = importlib.util.spec_from_file_location(
    "hierarchical_swarm_dashboard", _DASHBOARD_PATH
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
HierarchicalSwarmDashboard = _mod.HierarchicalSwarmDashboard


def _render(renderable, width: int = 200) -> str:
    buf = io.StringIO()
    console = Console(file=buf, width=width, highlight=False)
    console.print(renderable)
    return buf.getvalue()


def _make_orders(n: int) -> list:
    return [
        {"agent_name": f"Agent-{i}", "task": f"Task number {i}"}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# #1914 — director orders panel missing [:5] cap
# ---------------------------------------------------------------------------


class TestDirectorOrdersCap:
    def test_fewer_than_5_orders_all_shown(self):
        dash = HierarchicalSwarmDashboard()
        dash.director_orders = _make_orders(3)
        output = _render(dash._create_director_panel())
        for i in range(3):
            assert f"Agent-{i}" in output
        assert "more orders" not in output

    def test_exactly_5_orders_all_shown_no_truncation(self):
        dash = HierarchicalSwarmDashboard()
        dash.director_orders = _make_orders(5)
        output = _render(dash._create_director_panel())
        for i in range(5):
            assert f"Agent-{i}" in output
        assert "more orders" not in output

    def test_10_orders_only_first_5_shown(self):
        dash = HierarchicalSwarmDashboard()
        dash.director_orders = _make_orders(10)
        output = _render(dash._create_director_panel())

        for i in range(5):
            assert f"Agent-{i}" in output

        for i in range(5, 10):
            assert f"Agent-{i}" not in output

    def test_truncation_label_correct_count(self):
        dash = HierarchicalSwarmDashboard()
        dash.director_orders = _make_orders(10)
        output = _render(dash._create_director_panel())
        assert "5 more orders" in output

    def test_20_orders_truncation_label_says_15(self):
        dash = HierarchicalSwarmDashboard()
        dash.director_orders = _make_orders(20)
        output = _render(dash._create_director_panel())
        assert "15 more orders" in output
        for i in range(5, 20):
            assert f"Agent-{i}" not in output

    def test_zero_orders_shows_placeholder(self):
        dash = HierarchicalSwarmDashboard()
        dash.director_orders = []
        output = _render(dash._create_director_panel())
        assert "No orders available" in output
