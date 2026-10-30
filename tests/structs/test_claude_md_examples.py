import inspect
import pytest


def test_planner_worker_swarm_exported_from_swarms():
    """Verify PlannerWorkerSwarm can be imported directly from swarms."""
    from swarms import PlannerWorkerSwarm

    assert inspect.isclass(PlannerWorkerSwarm)


def test_council_as_a_judge_claude_md_instantiation():
    """Verify CouncilAsAJudge instantiation as documented in CLAUDE.md."""
    from swarms import CouncilAsAJudge

    council = CouncilAsAJudge(
        model_name="gpt-5.4",
        judge_agent_model_name="gpt-5.4",
        max_loops=1,
    )
    assert council.model_name == "gpt-5.4"


def test_debate_with_judge_claude_md_instantiation():
    """Verify DebateWithJudge instantiation as documented in CLAUDE.md."""
    from swarms import Agent, DebateWithJudge

    pro = Agent(
        agent_name="Pro",
        system_prompt="Argue strongly in favour.",
        model_name="gpt-5.4",
        max_loops=1,
    )
    con = Agent(
        agent_name="Con",
        system_prompt="Argue strongly against.",
        model_name="gpt-5.4",
        max_loops=1,
    )
    judge = Agent(
        agent_name="Judge",
        system_prompt="Evaluate debate.",
        model_name="gpt-5.4",
        max_loops=1,
    )

    debate = DebateWithJudge(
        pro_agent=pro,
        con_agent=con,
        judge_agent=judge,
        max_loops=3,
    )
    assert debate.pro_agent == pro
    assert debate.con_agent == con
    assert debate.judge_agent == judge


def test_heavy_swarm_claude_md_instantiation():
    """Verify HeavySwarm instantiation as documented in CLAUDE.md."""
    from swarms import HeavySwarm

    swarm = HeavySwarm(
        name="ResearchSwarm",
        question_agent_model_name="gpt-5.4",
        worker_model_name="gpt-5.4",
        max_loops=3,
        variant="heavy",
    )
    assert swarm.name == "ResearchSwarm"
