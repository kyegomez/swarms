"""Tests for HierarchicalSwarmDashboard dirty-panel refresh (issue #1677)."""

from unittest.mock import MagicMock, patch

from swarms.utils.hierarchical_swarm_dashboard import (
    HierarchicalSwarmDashboard,
)


def _started_dashboard() -> HierarchicalSwarmDashboard:
    dash = HierarchicalSwarmDashboard(swarm_name="Test-Swarm")
    dash.live_display = MagicMock()
    dash.is_active = True
    dash.max_loops = 3
    dash.start_time = 0.0
    dash._layout = dash._create_dashboard_layout()
    return dash


def test_layout_built_once_in_start():
    dash = HierarchicalSwarmDashboard(swarm_name="S")
    with patch.object(
        dash,
        "_create_dashboard_layout",
        wraps=dash._create_dashboard_layout,
    ) as mock:
        with patch("swarms.structs.hiearchical_swarm.Live"):
            dash.start(max_loops=1)
    assert mock.call_count == 1


def test_update_agent_status_rebuilds_only_agents():
    dash = _started_dashboard()
    with (
        patch.object(
            dash,
            "_create_agents_table",
            wraps=dash._create_agents_table,
        ) as agents,
        patch.object(
            dash,
            "_create_status_panel",
            wraps=dash._create_status_panel,
        ) as status,
        patch.object(
            dash,
            "_create_director_panel",
            wraps=dash._create_director_panel,
        ) as director,
    ):
        dash.update_agent_status("A", "RUNNING", task="t")
    assert agents.call_count == 1
    assert status.call_count == 0
    assert director.call_count == 0


def test_update_loop_rebuilds_only_status():
    dash = _started_dashboard()
    with (
        patch.object(
            dash,
            "_create_status_panel",
            wraps=dash._create_status_panel,
        ) as status,
        patch.object(
            dash,
            "_create_agents_table",
            wraps=dash._create_agents_table,
        ) as agents,
        patch.object(
            dash,
            "_create_director_panel",
            wraps=dash._create_director_panel,
        ) as director,
    ):
        dash.update_loop(2)
    assert status.call_count == 1
    assert agents.call_count == 0
    assert director.call_count == 0


def test_update_director_plan_rebuilds_only_director():
    dash = _started_dashboard()
    with (
        patch.object(
            dash,
            "_create_director_panel",
            wraps=dash._create_director_panel,
        ) as director,
        patch.object(
            dash,
            "_create_status_panel",
            wraps=dash._create_status_panel,
        ) as status,
        patch.object(
            dash,
            "_create_agents_table",
            wraps=dash._create_agents_table,
        ) as agents,
    ):
        dash.update_director_plan("plan")
    assert director.call_count == 1
    assert status.call_count == 0
    assert agents.call_count == 0


def test_update_director_orders_rebuilds_only_director():
    dash = _started_dashboard()
    with (
        patch.object(
            dash,
            "_create_director_panel",
            wraps=dash._create_director_panel,
        ) as director,
        patch.object(
            dash,
            "_create_status_panel",
            wraps=dash._create_status_panel,
        ) as status,
        patch.object(
            dash,
            "_create_agents_table",
            wraps=dash._create_agents_table,
        ) as agents,
    ):
        dash.update_director_orders(
            [{"agent_name": "A", "task": "t"}]
        )
    assert director.call_count == 1
    assert status.call_count == 0
    assert agents.call_count == 0


def test_force_refresh_rebuilds_all_panels():
    dash = _started_dashboard()
    with (
        patch.object(
            dash,
            "_create_status_panel",
            wraps=dash._create_status_panel,
        ) as status,
        patch.object(
            dash,
            "_create_director_panel",
            wraps=dash._create_director_panel,
        ) as director,
        patch.object(
            dash,
            "_create_agents_table",
            wraps=dash._create_agents_table,
        ) as agents,
    ):
        dash.force_refresh()
    assert status.call_count == 1
    assert director.call_count == 1
    assert agents.call_count == 1


def test_unknown_section_does_not_call_live_update():
    dash = _started_dashboard()
    dash._refresh_section("nonexistent_section")
    dash.live_display.update.assert_not_called()
