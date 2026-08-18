"""Tests for HierarchicalSwarmDashboard UI fixes.

Covers the bug fixed in PR #1916 / issue #1913:
  show_full_output() had a hardcoded width=120 on its Rich Panel —
  on terminals narrower than 120 chars the panel overflowed the viewport.

Uses importlib to load the dashboard module directly so the full
swarms package (and its heavy dependencies) are not triggered.
"""

import importlib.util
import io
import re
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


class _FakeLive:
    pass


def _dash_with_console(width: int):
    buf = io.StringIO()
    console = Console(file=buf, width=width, highlight=False)
    dash = HierarchicalSwarmDashboard()
    dash.console = console
    dash.live_display = _FakeLive()
    dash.is_active = True
    return dash, buf


def _plain(text: str) -> str:
    """Strip ANSI escape codes."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


# ---------------------------------------------------------------------------
# #1913 — show_full_output Panel hardcoded width=120
# ---------------------------------------------------------------------------


class TestShowFullOutputWidth:
    def test_panel_fits_80_char_terminal(self):
        dash, buf = _dash_with_console(80)
        dash.show_full_output("TestAgent", "Some output text")
        for line in buf.getvalue().splitlines():
            assert len(_plain(line)) <= 80, (
                f"Line exceeds 80 chars ({len(_plain(line))}): {_plain(line)!r}"
            )

    def test_panel_fits_100_char_terminal(self):
        dash, buf = _dash_with_console(100)
        dash.show_full_output("TestAgent", "Some output text")
        for line in buf.getvalue().splitlines():
            assert len(_plain(line)) <= 100

    def test_panel_fits_120_char_terminal(self):
        dash, buf = _dash_with_console(120)
        dash.show_full_output("TestAgent", "Some output text")
        for line in buf.getvalue().splitlines():
            assert len(_plain(line)) <= 120

    def test_panel_renders_agent_name(self):
        dash, buf = _dash_with_console(120)
        dash.show_full_output("MyAgent", "Hello from agent")
        assert "MyAgent" in buf.getvalue()

    def test_panel_renders_output_content(self):
        dash, buf = _dash_with_console(120)
        dash.show_full_output("MyAgent", "Hello from agent")
        assert "Hello from agent" in buf.getvalue()

    def test_no_output_when_inactive(self):
        """show_full_output must be silent when is_active=False."""
        buf = io.StringIO()
        console = Console(file=buf, width=80, highlight=False)
        dash = HierarchicalSwarmDashboard()
        dash.console = console
        dash.is_active = False

        dash.show_full_output("SilentAgent", "Should not appear")
        assert buf.getvalue() == ""
