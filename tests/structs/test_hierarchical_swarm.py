import os
from typing import Any

import pytest

from swarms import Agent
from swarms.schemas.hs_schemas import OrderBatch
from swarms.structs.hiearchical_swarm import (
    HierarchicalOrder,
    HierarchicalSwarm,
)
from swarms.utils.workspace_utils import get_workspace_dir


def test_hierarchical_swarm_basic_initialization():
    """Test basic HierarchicalSwarm initialization"""
    # Create worker agents
    research_agent = Agent(
        agent_name="Research-Specialist",
        agent_description="Specialist in research and data collection",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    analysis_agent = Agent(
        agent_name="Analysis-Expert",
        agent_description="Expert in data analysis and insights",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    implementation_agent = Agent(
        agent_name="Implementation-Manager",
        agent_description="Manager for implementation and execution",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    # Create swarm with agents
    swarm = HierarchicalSwarm(
        name="Research-Analysis-Implementation-Swarm",
        description="Hierarchical swarm for comprehensive project execution",
        agents=[research_agent, analysis_agent, implementation_agent],
        max_loops=1,
    )

    # Verify initialization
    assert swarm.name == "Research-Analysis-Implementation-Swarm"
    assert (
        swarm.description
        == "Hierarchical swarm for comprehensive project execution"
    )
    assert len(swarm.agents) == 3
    assert swarm.max_loops == 1
    assert swarm.director is not None


def test_hierarchical_swarm_with_director():
    """Test HierarchicalSwarm with custom director"""
    # Create a custom director
    director = Agent(
        agent_name="Project-Director",
        agent_description="Senior project director with extensive experience",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    # Create worker agents
    developer = Agent(
        agent_name="Senior-Developer",
        agent_description="Senior software developer",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    tester = Agent(
        agent_name="QA-Lead",
        agent_description="Quality assurance lead",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    # Create swarm with custom director
    swarm = HierarchicalSwarm(
        name="Software-Development-Swarm",
        description="Hierarchical swarm for software development projects",
        director=director,
        agents=[developer, tester],
        max_loops=2,
    )

    assert swarm.director == director
    assert len(swarm.agents) == 2
    assert swarm.max_loops == 2


def test_hierarchical_swarm_execution():
    """Test HierarchicalSwarm execution with multiple agents"""
    # Create specialized agents
    market_researcher = Agent(
        agent_name="Market-Researcher",
        agent_description="Market research specialist",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    product_strategist = Agent(
        agent_name="Product-Strategist",
        agent_description="Product strategy and planning expert",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    technical_architect = Agent(
        agent_name="Technical-Architect",
        agent_description="Technical architecture and design specialist",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    risk_analyst = Agent(
        agent_name="Risk-Analyst",
        agent_description="Risk assessment and mitigation specialist",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    # Create hierarchical swarm
    swarm = HierarchicalSwarm(
        name="Product-Development-Swarm",
        description="Comprehensive product development hierarchical swarm",
        agents=[
            market_researcher,
            product_strategist,
            technical_architect,
            risk_analyst,
        ],
        max_loops=1,
        verbose=True,
    )

    # Execute swarm
    result = swarm.run(
        "Develop a comprehensive strategy for a new AI-powered healthcare platform"
    )

    # Verify result structure
    assert result is not None
    # HierarchicalSwarm returns a SwarmSpec or conversation history, just ensure it's not None


def test_hierarchical_swarm_multiple_loops():
    """Test HierarchicalSwarm with multiple feedback loops"""
    # Create agents for iterative refinement
    planner = Agent(
        agent_name="Strategic-Planner",
        agent_description="Strategic planning and project management",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    executor = Agent(
        agent_name="Task-Executor",
        agent_description="Task execution and implementation",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    reviewer = Agent(
        agent_name="Quality-Reviewer",
        agent_description="Quality assurance and review specialist",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    # Create swarm with multiple loops for iterative refinement
    swarm = HierarchicalSwarm(
        name="Iterative-Development-Swarm",
        description="Hierarchical swarm with iterative feedback loops",
        agents=[planner, executor, reviewer],
        max_loops=3,  # Allow multiple iterations
        verbose=True,
    )

    # Execute with multiple loops
    result = swarm.run(
        "Create a detailed project plan for implementing a machine learning recommendation system"
    )

    assert result is not None


def test_hierarchical_swarm_error_handling():
    """Test HierarchicalSwarm error handling"""
    # Test with empty agents list
    try:
        HierarchicalSwarm(agents=[])
        assert (
            False
        ), "Should have raised ValueError for empty agents list"
    except ValueError as e:
        assert "agents" in str(e).lower() or "empty" in str(e).lower()

    # Test with invalid max_loops
    researcher = Agent(
        agent_name="Test-Researcher",
        agent_description="Test researcher",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    try:
        HierarchicalSwarm(agents=[researcher], max_loops=0)
        assert (
            False
        ), "Should have raised ValueError for invalid max_loops"
    except ValueError as e:
        assert "max_loops" in str(e).lower() or "0" in str(e)


def test_hierarchical_swarm_collaboration_prompts():
    """Test HierarchicalSwarm with collaboration prompts enabled"""
    # Create agents
    data_analyst = Agent(
        agent_name="Data-Analyst",
        agent_description="Data analysis specialist",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    business_analyst = Agent(
        agent_name="Business-Analyst",
        agent_description="Business analysis specialist",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    # Create swarm with collaboration prompts
    swarm = HierarchicalSwarm(
        name="Collaborative-Analysis-Swarm",
        description="Hierarchical swarm with enhanced collaboration",
        agents=[data_analyst, business_analyst],
        max_loops=1,
        add_collaboration_prompt=True,
    )

    # Check that collaboration prompts were added to agents
    assert data_analyst.system_prompt is not None
    assert business_analyst.system_prompt is not None

    # Execute swarm
    result = swarm.run(
        "Analyze customer behavior patterns and provide business recommendations"
    )
    assert result is not None


def test_hierarchical_swarm_real_world_scenario():
    """Test HierarchicalSwarm in a realistic business scenario"""
    # Create agents representing different business functions
    market_intelligence = Agent(
        agent_name="Market-Intelligence-Director",
        agent_description="Director of market intelligence and competitive analysis",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    product_strategy = Agent(
        agent_name="Product-Strategy-Manager",
        agent_description="Product strategy and roadmap manager",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    engineering_lead = Agent(
        agent_name="Engineering-Lead",
        agent_description="Senior engineering lead and technical architect",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    operations_manager = Agent(
        agent_name="Operations-Manager",
        agent_description="Operations and implementation manager",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    compliance_officer = Agent(
        agent_name="Compliance-Officer",
        agent_description="Legal compliance and regulatory specialist",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    # Create comprehensive hierarchical swarm
    swarm = HierarchicalSwarm(
        name="Enterprise-Strategy-Swarm",
        description="Enterprise-level strategic planning and execution swarm",
        agents=[
            market_intelligence,
            product_strategy,
            engineering_lead,
            operations_manager,
            compliance_officer,
        ],
        max_loops=2,
        verbose=True,
        add_collaboration_prompt=True,
    )

    # Test with complex enterprise scenario
    result = swarm.run(
        "Develop a comprehensive 5-year strategic plan for our company to become a leader in "
        "AI-powered enterprise solutions. Consider market opportunities, competitive landscape, "
        "technical requirements, operational capabilities, and regulatory compliance."
    )

    assert result is not None


def test_hierarchical_swarm_autosave_creates_workspace_dir(
    monkeypatch, tmp_path
):
    """Test that HierarchicalSwarm with autosave=True creates a workspace directory."""
    get_workspace_dir.cache_clear()
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))

    agent1 = Agent(
        agent_name="Autosave-Hierarchical-1",
        agent_description="Agent for autosave test",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )
    agent2 = Agent(
        agent_name="Autosave-Hierarchical-2",
        agent_description="Agent for autosave test",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    swarm = HierarchicalSwarm(
        name="Autosave-Test-Swarm",
        description="Hierarchical swarm for autosave test",
        agents=[agent1, agent2],
        max_loops=1,
        autosave=True,
        verbose=False,
    )

    assert swarm.autosave is True
    assert swarm.swarm_workspace_dir is not None
    assert os.path.isdir(swarm.swarm_workspace_dir)
    assert "HierarchicalSwarm" in swarm.swarm_workspace_dir
    assert "Autosave-Test-Swarm" in swarm.swarm_workspace_dir

    get_workspace_dir.cache_clear()


def test_hierarchical_swarm_autosave_saves_conversation_after_run(
    monkeypatch, tmp_path
):
    """Test that HierarchicalSwarm saves conversation_history.json after run when autosave=True."""
    get_workspace_dir.cache_clear()
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))

    agent1 = Agent(
        agent_name="Autosave-Run-Hierarchical-1",
        agent_description="Agent for autosave run test",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )
    agent2 = Agent(
        agent_name="Autosave-Run-Hierarchical-2",
        agent_description="Agent for autosave run test",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    swarm = HierarchicalSwarm(
        name="Autosave-Run-Swarm",
        description="Hierarchical swarm for autosave run test",
        agents=[agent1, agent2],
        max_loops=1,
        autosave=True,
        verbose=False,
    )

    result = swarm.run(task="Say hello in one short sentence.")
    assert result is not None

    conversation_path = os.path.join(
        swarm.swarm_workspace_dir, "conversation_history.json"
    )
    assert os.path.isfile(
        conversation_path
    ), f"Expected conversation_history.json at {conversation_path}"

    get_workspace_dir.cache_clear()


##############################################################################
# Director settings and worker recovery tests
##############################################################################


class StubAgent:
    def __init__(
        self,
        agent_name: str,
        outputs: list[Any],
    ):
        self.agent_name = agent_name
        self.description = f"{agent_name} test agent"
        self.system_prompt = self.description
        self.outputs = iter(outputs)
        self.calls = 0
        self.output_type = "dict"

    def run(self, *args, **kwargs):
        self.calls += 1
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return output


def make_recovery_swarm(
    director: StubAgent,
    workers: list[StubAgent],
    **kwargs,
) -> HierarchicalSwarm:
    return HierarchicalSwarm(
        director=director,
        agents=workers,
        autosave=False,
        planning_enabled=False,
        director_feedback_on=False,
        add_collaboration_prompt=False,
        **kwargs,
    )


def test_director_settings_are_forwarded(monkeypatch):
    captured = {}

    def build_director(**kwargs):
        captured.update(kwargs)
        return StubAgent(kwargs["agent_name"], [])

    monkeypatch.setattr(
        "swarms.structs.hiearchical_swarm.Agent",
        build_director,
    )
    worker = StubAgent("Worker", ["done"])

    swarm = HierarchicalSwarm(
        agents=[worker],
        autosave=False,
        planning_enabled=False,
        director_settings={
            "agent_name": "Custom Director",
            "model_name": "custom-model",
            "max_loops": 3,
            "reasoning_effort": "high",
            "output_type": "dict",
        },
    )

    assert swarm.director_name == "Custom Director"
    assert swarm.director_model_name == "custom-model"
    assert captured["max_loops"] == 3
    assert captured["reasoning_effort"] == "high"
    assert captured["output_type"] == "final"
    assert captured["base_model"] is OrderBatch
    assert worker.output_type == "final"
    assert swarm.director.output_type == "final"


def test_custom_director_and_workers_are_forced_to_final_output():
    director = StubAgent("Director", [])
    worker = StubAgent("Worker", ["done"])

    swarm = make_recovery_swarm(director, [worker])

    assert director.output_type == "final"
    assert worker.output_type == "final"

    plan, orders = swarm.parse_orders(
        '{"plan": "Use the worker", "orders": '
        '[{"agent_name": "Worker", "task": "Do the work"}]}'
    )
    assert plan == "Use the worker"
    assert orders[0].agent_name == "Worker"


def test_step_skips_feedback_on_final_loop(monkeypatch):
    swarm = make_recovery_swarm(
        StubAgent("Director", []),
        [StubAgent("Worker", [])],
        max_loops=2,
    )
    swarm.director_feedback_on = True
    feedback_calls = []
    director_output = {
        "orders": [{"agent_name": "Worker", "task": "Do the work"}],
    }

    monkeypatch.setattr(
        swarm, "run_director", lambda task, img=None: director_output
    )
    monkeypatch.setattr(
        swarm,
        "execute_orders",
        lambda orders: ["done"],
    )
    monkeypatch.setattr(
        swarm,
        "feedback_director",
        lambda outputs: feedback_calls.append(outputs) or "feedback",
    )

    assert swarm.step("task") == "feedback"
    assert swarm.step("task", is_final_loop=True) == ["done"]
    assert feedback_calls == [["done"]]


def test_step_returns_immediately_when_director_has_no_orders(
    monkeypatch,
):
    swarm = make_recovery_swarm(
        StubAgent("Director", []),
        [StubAgent("Worker", [])],
    )
    monkeypatch.setattr(
        swarm,
        "run_director",
        lambda task, img=None: {
            "plan": "Nothing needed",
            "orders": [],
        },
    )

    def unexpected_execution(*args, **kwargs):
        raise AssertionError("empty plans must not execute workers")

    monkeypatch.setattr(swarm, "execute_orders", unexpected_execution)

    assert swarm.step("task") == []


def test_run_marks_only_the_last_step_as_final(monkeypatch):
    swarm = make_recovery_swarm(
        StubAgent("Director", []),
        [StubAgent("Worker", [])],
        max_loops=2,
    )
    final_flags = []

    def capture_step(*args, is_final_loop=False, **kwargs):
        final_flags.append(is_final_loop)
        return []

    monkeypatch.setattr(swarm, "step", capture_step)

    swarm.run("task")

    assert final_flags == [False, True]


def test_batched_run_forwards_utility_options(monkeypatch):
    swarm = make_recovery_swarm(
        StubAgent("Director", []),
        [StubAgent("Worker", [])],
    )
    captured = {}

    def fake_batched_run(func, tasks, *args, **kwargs):
        captured.update(
            func=func,
            tasks=tasks,
            args=args,
            kwargs=kwargs,
        )
        return ["first", "second"]

    monkeypatch.setattr(
        "swarms.structs.hiearchical_swarm.batched_run",
        fake_batched_run,
    )

    result = swarm.batched_run(
        ["task-1", "task-2"],
        "positional",
        imgs=["one.png", "two.png"],
        max_workers=2,
        return_agent_output_dict=True,
        return_exceptions=True,
    )

    assert result == ["first", "second"]
    assert captured["func"] == swarm.run
    assert captured["tasks"] == ["task-1", "task-2"]
    assert captured["args"] == ("positional",)
    assert captured["kwargs"] == {
        "img": None,
        "imgs": ["one.png", "two.png"],
        "max_workers": 2,
        "return_agent_output_dict": True,
        "return_exceptions": True,
    }


def test_failed_worker_is_retried_and_reassigned():
    failed_worker = StubAgent(
        "Failed Worker",
        [RuntimeError("offline"), RuntimeError("offline")],
    )
    healthy_worker = StubAgent("Healthy Worker", ["recovered"])
    director = StubAgent(
        "Director",
        [
            {
                "plan": "Move the failed task to a healthy worker.",
                "orders": [
                    {
                        "agent_name": "Healthy Worker",
                        "task": "complete the task",
                    }
                ],
            }
        ],
    )
    swarm = make_recovery_swarm(
        director,
        [failed_worker, healthy_worker],
        max_agent_retries=1,
        max_reassignment_attempts=1,
        parallel_execution=False,
    )

    outputs = swarm.execute_orders(
        [
            HierarchicalOrder(
                agent_name="Failed Worker",
                task="complete the task",
            )
        ]
    )

    assert failed_worker.calls == 2
    assert healthy_worker.calls == 1
    assert director.calls == 1
    assert outputs[-1] == "recovered"
    assert "[WORKER UNAVAILABLE]" in swarm.conversation.get_str()
    assert "[RECOVERY STARTED]" in swarm.conversation.get_str()


def test_one_failed_worker_does_not_stop_other_orders():
    failed_worker = StubAgent(
        "Failed Worker",
        [RuntimeError("offline")],
    )
    healthy_worker = StubAgent("Healthy Worker", ["completed"])
    director = StubAgent("Director", [])
    swarm = make_recovery_swarm(
        director,
        [failed_worker, healthy_worker],
        max_agent_retries=0,
        max_reassignment_attempts=0,
        parallel_execution=True,
    )

    outputs = swarm.execute_orders(
        [
            HierarchicalOrder(
                agent_name="Failed Worker",
                task="first task",
            ),
            HierarchicalOrder(
                agent_name="Healthy Worker",
                task="second task",
            ),
        ]
    )

    assert outputs[0]["status"] == "failed"
    assert outputs[1] == "completed"
    assert healthy_worker.calls == 1


##############################################################################
# max_workers and nested-swarm worker tests
##############################################################################


class StubNestedSwarm:
    """Mimic a nested orchestrator identified by ``name``."""

    def __init__(self, name: str, outputs: list):
        self.name = name
        self.description = f"{name} nested swarm"
        self.outputs = iter(outputs)
        self.calls = 0
        self.output_type = "dict-all-except-first"

    def run(self, task, *args, **kwargs):
        self.calls += 1
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return output


class StubFlatOrchestrator:
    """Mimic an orchestrator with a strict ``run`` signature."""

    def __init__(self, name: str, outputs: list):
        self.name = name
        self.description = f"{name} orchestrator"
        self.outputs = iter(outputs)
        self.calls = 0
        self.output_type = "dict"

    def run(self, task, img=None):
        self.calls += 1
        return next(self.outputs)


def test_max_workers_defaults_to_cpu_heuristic():
    swarm = make_recovery_swarm(
        StubAgent("Director", []),
        [StubAgent("Worker", ["done"])],
    )
    assert swarm.max_workers == max(
        1, int((os.cpu_count() or 1) * 0.95)
    )


def test_max_workers_override_is_used():
    swarm = make_recovery_swarm(
        StubAgent("Director", []),
        [StubAgent("Worker", ["done"])],
        max_workers=2,
    )
    assert swarm.max_workers == 2


def test_max_workers_must_be_positive():
    with pytest.raises(ValueError):
        make_recovery_swarm(
            StubAgent("Director", []),
            [StubAgent("Worker", ["done"])],
            max_workers=0,
        )


def test_nested_swarm_worker_is_called():
    sub_team = StubNestedSwarm("SubTeam", ["sub-team result"])
    director = StubAgent("Director", [])
    swarm = make_recovery_swarm(
        director,
        [sub_team],
        parallel_execution=False,
    )

    output = swarm.call_single_agent(
        "SubTeam",
        "handle part A",
    )

    assert output == "sub-team result"
    assert sub_team.calls == 1


def test_orchestrator_with_strict_run_signature_completes():
    flat = StubFlatOrchestrator("FlatOrchestrator", ["moa result"])
    director = StubAgent("Director", [])
    swarm = make_recovery_swarm(
        director,
        [flat],
        parallel_execution=False,
    )

    output = swarm.call_single_agent(
        "FlatOrchestrator",
        "handle part B",
    )

    assert output == "moa result"
    assert flat.calls == 1


def test_reassignment_targets_nested_swarm_worker_by_name():
    failed_worker = StubAgent(
        "Failed Worker",
        [RuntimeError("offline"), RuntimeError("offline")],
    )
    healthy_sub_team = StubNestedSwarm(
        "Healthy SubTeam", ["recovered by subteam"]
    )
    director = StubAgent(
        "Director",
        [
            {
                "plan": "Move the failed task to the healthy nested sub-team.",
                "orders": [
                    {
                        "agent_name": "Healthy SubTeam",
                        "task": "complete the task",
                    }
                ],
            }
        ],
    )
    swarm = make_recovery_swarm(
        director,
        [failed_worker, healthy_sub_team],
        max_agent_retries=1,
        max_reassignment_attempts=1,
        parallel_execution=False,
    )

    outputs = swarm.execute_orders(
        [
            HierarchicalOrder(
                agent_name="Failed Worker",
                task="complete the task",
            )
        ]
    )

    assert healthy_sub_team.calls == 1
    assert outputs[-1] == "recovered by subteam"
    assert "[RECOVERY NOTICE]" not in swarm.conversation.get_str()


# ============================================================================
# Context management: how much each agent is re-sent
# ============================================================================


def _scripted_hs_agent(name, seen, director=False):
    """A real Agent with a stubbed LLM call, so short_memory is real."""
    import json as _json

    from swarms import Agent

    agent = Agent(
        agent_name=name,
        model_name="gpt-4o-mini",
        max_loops=1,
        persistent_memory=False,
        print_on=False,
        autosave=False,
    )

    def fake_call_llm(task=None, *args, **kwargs):
        messages = kwargs.get("messages") or []
        seen.append((name, messages))
        if director:
            return [
                {
                    "id": "d1",
                    "type": "function",
                    "function": {
                        "name": "handoff",
                        "arguments": _json.dumps(
                            {
                                "plan": "do it",
                                "orders": [
                                    {
                                        "agent_name": "W1",
                                        "task": "do part 1",
                                    }
                                ],
                            }
                        ),
                    },
                }
            ]
        return f"[{name}-out]"

    agent.call_llm = fake_call_llm
    return agent


class TestHierarchicalContextManagement:
    """
    Every agent used to receive ``History: <entire conversation>`` on each
    invocation. That history lands in the agent's own memory, so the next
    invocation re-sent all of it on top - the director's prompt grew about
    sevenfold across two loops.
    """

    def _run_two_loops(self):
        from swarms import HierarchicalSwarm

        seen = []
        swarm = HierarchicalSwarm(
            director=_scripted_hs_agent(
                "Director", seen, director=True
            ),
            agents=[_scripted_hs_agent("W1", seen)],
            max_loops=2,
        )
        swarm.run("Build something.")
        return swarm, seen

    def test_the_director_prompt_does_not_balloon(self):
        _, seen = self._run_two_loops()
        director_turns = [
            sum(len(str(m.get("content"))) for m in messages)
            for name, messages in seen
            if name == "Director"
        ]
        assert len(director_turns) >= 2, "the director ran only once"

        first, second = director_turns[0], director_turns[1]
        assert second < first * 5, (
            f"director context grew {second / max(first, 1):.1f}x "
            f"across one loop: {director_turns}"
        )

    def test_a_worker_sees_its_own_output_once_as_assistant(self):
        """The compounding this guards against is the agent re-reading itself.

        Typed turns carry the whole conversation every request, so raw growth
        is linear and no longer the signal. What must not happen is the
        worker's own output coming back a second time, mislabelled as
        something someone else said - that is what compounded across loops.
        """
        _, seen = self._run_two_loops()
        worker_calls = [
            messages for name, messages in seen if name == "W1"
        ]
        if len(worker_calls) < 2:
            return

        second = worker_calls[1]
        # Turns that are the output, not ones quoting it inside a larger
        # blob: the feedback director still interpolates a flattened history
        # (#2033), which is a separate unconverted path.
        own = [
            message
            for message in second
            if str(message.get("content")).strip() == "[W1-out]"
        ]
        assert (
            len(own) == 1
        ), f"the worker saw its own output {len(own)} times: {own}"
        assert (
            own[0]["role"] == "assistant"
        ), f"own output arrived as {own[0]['role']!r}, not 'assistant'"

    def test_the_director_tool_call_is_stored_readably(self):
        """
        A raw tool-call list renders as a Python repr in the history that
        every later agent then has to read.
        """
        swarm, _ = self._run_two_loops()
        director_messages = [
            str(m["content"])
            for m in swarm.conversation.conversation_history
            if m["role"] == "Director"
        ]
        assert director_messages, "the director recorded nothing"
        assert not any(
            c.startswith("[{'id'") for c in director_messages
        ), "the director's tool call was stored as a Python repr"

    def test_the_conversation_starts_clean(self):
        from swarms import HierarchicalSwarm

        seen = []
        swarm = HierarchicalSwarm(
            director=_scripted_hs_agent(
                "Director", seen, director=True
            ),
            agents=[_scripted_hs_agent("W1", seen)],
            max_loops=1,
        )
        roles = [
            m["role"] for m in swarm.conversation.conversation_history
        ]
        assert (
            "user" not in roles
        ), f"stale messages loaded from disk: {roles}"


def _swarm_with_one_order(print_on):
    """A swarm whose director always issues a single order."""
    seen = []
    return HierarchicalSwarm(
        director=_scripted_hs_agent("Director", seen, director=True),
        agents=[_scripted_hs_agent("W1", seen)],
        max_loops=1,
        print_on=print_on,
    )


def test_the_director_panel_prints_without_verbose(capsys):
    swarm = _swarm_with_one_order(print_on=True)
    assert swarm.verbose is False

    swarm.step("Build something.")

    printed = capsys.readouterr().out
    assert "Director Name: Director" in printed
    assert "W1" in printed


def test_print_on_false_silences_the_director_panel(capsys):
    swarm = _swarm_with_one_order(print_on=False)

    swarm.step("Build something.")

    assert "Director Name" not in capsys.readouterr().out


def test_the_director_panel_carries_the_plan(capsys):
    swarm = _swarm_with_one_order(print_on=True)

    swarm.step("Build something.")

    printed = capsys.readouterr().out
    assert "Plan" in printed
    # the scripted director's plan, from _scripted_hs_agent
    assert "do it" in printed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_second_run_does_not_carry_the_first_task():
    """A reused swarm starts each task from an empty conversation and cursor."""
    director = StubAgent(
        "Director",
        [
            OrderBatch(
                orders=[
                    HierarchicalOrder(
                        agent_name="Worker", task="first subtask"
                    )
                ]
            ),
            OrderBatch(
                orders=[
                    HierarchicalOrder(
                        agent_name="Worker", task="second subtask"
                    )
                ]
            ),
        ],
    )
    worker = StubAgent("Worker", ["first answer", "second answer"])
    swarm = make_recovery_swarm(director, [worker], max_loops=1)

    swarm.run("first task")
    swarm.run("second task")

    history = swarm.conversation.get_str()

    assert "first task" not in history, history
    assert "first answer" not in history, history
    assert "second task" in history, history
    assert swarm._delivered.get("Worker", 0) <= len(
        swarm.conversation.conversation_history
    )
