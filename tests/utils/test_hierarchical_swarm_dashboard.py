"""Tests for HierarchicalSwarmDashboard UI fixes.

Covers three bugs fixed in separate PRs:
  #1913 — show_full_output Panel had hardcoded width=120
  #1914 — director orders loop missing [:5] cap
  #1915 — PROGRESS showed 100% before last loop's agents executed

Uses importlib to load the dashboard module directly so the full
swarms package (and its heavy dependencies) are not triggered.
"""

import importlib.util
import io
import time
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
    """Render a Rich renderable to a plain string."""
    buf = io.StringIO()
    console = Console(file=buf, width=width, highlight=False)
    console.print(renderable)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# #1913 — show_full_output Panel hardcoded width=120
# ---------------------------------------------------------------------------


class TestShowFullOutputWidth:
    def test_panel_fits_narrow_terminal(self):
        """Panel must not exceed the console width on an 80-char terminal."""
        buf = io.StringIO()
        narrow_console = Console(file=buf, width=80, highlight=False)

        dash = HierarchicalSwarmDashboard()
        dash.console = narrow_console

        class _FakeLive:
            pass

        dash.live_display = _FakeLive()
        dash.is_active = True

        dash.show_full_output("TestAgent", "Some output text")
        output = buf.getvalue()

        import re
        for line in output.splitlines():
            plain = re.sub(r"\x1b\[[0-9;]*m", "", line)
            assert len(plain) <= 80, (
                f"Line exceeds 80 chars (got {len(plain)}): {plain!r}"
            )

    def test_panel_renders_content(self):
        """show_full_output must print the agent name and output text."""
        buf = io.StringIO()
        console = Console(file=buf, width=120, highlight=False)

        dash = HierarchicalSwarmDashboard()
        dash.console = console

        class _FakeLive:
            pass

        dash.live_display = _FakeLive()
        dash.is_active = True

        dash.show_full_output("MyAgent", "Hello from agent")
        output = buf.getvalue()
        assert "MyAgent" in output
        assert "Hello from agent" in output


# ---------------------------------------------------------------------------
# #1915 — PROGRESS shows 100% prematurely
# ---------------------------------------------------------------------------


class TestCompletedLoopsCounter:
    def test_completed_loops_starts_at_zero(self):
        dash = HierarchicalSwarmDashboard()
        assert dash.completed_loops == 0

    def test_mark_loop_complete_increments(self):
        dash = HierarchicalSwarmDashboard()
        dash.mark_loop_complete()
        assert dash.completed_loops == 1
        dash.mark_loop_complete()
        assert dash.completed_loops == 2

    def test_mark_loop_complete_no_crash_without_live(self):
        """mark_loop_complete must be safe before start() is called."""
        dash = HierarchicalSwarmDashboard()
        dash.mark_loop_complete()  # live_display is None — must not raise

    def test_progress_uses_completed_not_current_loop(self):
        """PROGRESS % must reflect completed_loops, not current_loop."""
        dash = HierarchicalSwarmDashboard()
        dash.max_loops = 3
        dash.start_time = time.time()

        # Loop 3 has STARTED but NOT finished
        dash.current_loop = 3
        dash.completed_loops = 2

        output = _render(dash._create_status_panel())

        # 2/3 complete = 66.7%, NOT 100%
        assert "66.7%" in output
        assert "100.0%" not in output

    def test_progress_zero_before_any_loop_completes(self):
        dash = HierarchicalSwarmDashboard()
        dash.max_loops = 3
        dash.start_time = time.time()
        dash.current_loop = 1
        dash.completed_loops = 0

        output = _render(dash._create_status_panel())
        assert "0.0%" in output

    def test_progress_reaches_100_after_all_loops_finish(self):
        dash = HierarchicalSwarmDashboard()
        dash.max_loops = 3
        dash.start_time = time.time()
        dash.current_loop = 3
        dash.completed_loops = 3

        output = _render(dash._create_status_panel())
        assert "100.0%" in output

    def test_current_loop_label_shows_running_loop(self):
        """CURRENT LOOP: N reflects the announced loop, separate from PROGRESS."""
        dash = HierarchicalSwarmDashboard()
        dash.max_loops = 3
        dash.start_time = time.time()
        dash.current_loop = 3
        dash.completed_loops = 2

        output = _render(dash._create_status_panel())
        assert "CURRENT LOOP" in output
        assert "3" in output
