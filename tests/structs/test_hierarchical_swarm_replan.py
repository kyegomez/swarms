"""
Tests for HierarchicalSwarm incremental replan on judge rejection.
"""

from unittest.mock import MagicMock, patch

from swarms.structs.hiearchical_swarm import (
    AgentScore,
    HierarchicalOrder,
    HierarchicalSwarm,
    JudgeReport,
    ReplanAction,
    ReplanActionType,
)


def _make_swarm(**kwargs) -> HierarchicalSwarm:
    worker = MagicMock()
    worker.agent_name = "WorkerA"
    defaults = dict(
        agents=[worker],
        autosave=False,
        agent_as_judge=True,
        verbose=False,
        interactive=False,
    )
    defaults.update(kwargs)
    with patch.object(
        HierarchicalSwarm, "init_swarm", return_value=None
    ):
        swarm = HierarchicalSwarm(**defaults)
    swarm.conversation = MagicMock()
    swarm.conversation.get_str.return_value = ""
    return swarm


def _revise_report(failed: list) -> str:
    return JudgeReport(
        overall_quality=3,
        scores=[
            AgentScore(
                agent_name=n, score=3, reasoning="r", suggestions="s"
            )
            for n in failed
        ],
        summary="needs work",
        verdict="REVISE",
        feedback="Outputs were incomplete.",
        failed_subtasks=failed,
    ).model_dump_json()


def _approve_report() -> str:
    return JudgeReport(
        overall_quality=9,
        scores=[
            AgentScore(
                agent_name="WorkerA",
                score=9,
                reasoning="r",
                suggestions="s",
            )
        ],
        summary="great",
        verdict="APPROVE",
    ).model_dump_json()


# -- _parse_judge_report ------------------------------------------------------


def test_parse_judge_report_from_json_string():
    swarm = _make_swarm()
    report = swarm._parse_judge_report(_revise_report(["WorkerA"]))
    assert report is not None
    assert report.verdict == "REVISE"
    assert report.failed_subtasks == ["WorkerA"]


def test_parse_judge_report_returns_none_on_garbage():
    swarm = _make_swarm()
    assert swarm._parse_judge_report("not json {{{{") is None


# -- _replan ------------------------------------------------------------------


def test_replan_returns_orders_on_success():
    swarm = _make_swarm()
    swarm.list_worker_agents = MagicMock(return_value="WorkerA")

    replan_action = ReplanAction(
        action_type=ReplanActionType.REASSIGN,
        orders=[
            HierarchicalOrder(
                agent_name="WorkerA", task="Retry", context=""
            )
        ],
        reasoning="failed",
    ).model_dump_json()

    mock_agent = MagicMock()
    mock_agent.run.return_value = replan_action

    with patch(
        "swarms.structs.hiearchical_swarm.Agent",
        return_value=mock_agent,
    ):
        orders = swarm._replan(
            "task", "bad output", ["WorkerA"], {"WorkerA": "bad"}
        )

    assert len(orders) == 1
    assert orders[0].agent_name == "WorkerA"


def test_replan_returns_empty_on_failure():
    swarm = _make_swarm()
    swarm.list_worker_agents = MagicMock(return_value="WorkerA")

    with patch(
        "swarms.structs.hiearchical_swarm.Agent",
        side_effect=RuntimeError("down"),
    ):
        orders = swarm._replan("task", "bad", ["WorkerA"], {})

    assert orders == []


# -- step() -------------------------------------------------------------------


def test_step_triggers_replan_on_revise():
    swarm = _make_swarm()
    swarm.run_director = MagicMock(return_value={})
    swarm.list_worker_agents = MagicMock(return_value="WorkerA")

    order = HierarchicalOrder(
        agent_name="WorkerA", task="t", context=""
    )
    swarm.parse_orders = MagicMock(return_value=("plan", [order]))
    swarm.execute_orders = MagicMock(side_effect=[["bad"], ["good"]])
    swarm.run_judge_agent = MagicMock(
        return_value=_revise_report(["WorkerA"])
    )
    swarm._replan = MagicMock(return_value=[order])

    swarm.step(task="do something")

    swarm._replan.assert_called_once()
    assert swarm.execute_orders.call_count == 2


def test_step_skips_replan_on_approve():
    swarm = _make_swarm()
    swarm.run_director = MagicMock(return_value={})

    order = HierarchicalOrder(
        agent_name="WorkerA", task="t", context=""
    )
    swarm.parse_orders = MagicMock(return_value=("plan", [order]))
    swarm.execute_orders = MagicMock(return_value=["good"])
    swarm.run_judge_agent = MagicMock(return_value=_approve_report())
    swarm._replan = MagicMock()

    swarm.step(task="do something")

    swarm._replan.assert_not_called()


def test_step_merges_outputs_after_replan():
    swarm = _make_swarm()
    worker_b = MagicMock()
    worker_b.agent_name = "WorkerB"
    swarm.agents.append(worker_b)
    swarm.run_director = MagicMock(return_value={})
    swarm.list_worker_agents = MagicMock(
        return_value="WorkerA, WorkerB"
    )

    orders = [
        HierarchicalOrder(
            agent_name="WorkerA", task="t1", context=""
        ),
        HierarchicalOrder(
            agent_name="WorkerB", task="t2", context=""
        ),
    ]
    replan_order = HierarchicalOrder(
        agent_name="WorkerA", task="retry", context=""
    )

    swarm.parse_orders = MagicMock(return_value=("plan", orders))
    swarm.execute_orders = MagicMock(
        side_effect=[["bad_A", "good_B"], ["fixed_A"]]
    )
    swarm.run_judge_agent = MagicMock(
        return_value=_revise_report(["WorkerA"])
    )
    swarm._replan = MagicMock(return_value=[replan_order])

    result = swarm.step(task="do something")

    assert "fixed_A" in result
    assert "good_B" in result
