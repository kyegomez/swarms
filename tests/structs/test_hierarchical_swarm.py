import os
from typing import Any

import pytest

from swarms import Agent
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


def test_hierarchical_swarm_with_dashboard():
    """Test HierarchicalSwarm with interactive dashboard"""
    # Create agents
    content_creator = Agent(
        agent_name="Content-Creator",
        agent_description="Content creation specialist",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    editor = Agent(
        agent_name="Editor",
        agent_description="Content editor and proofreader",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    publisher = Agent(
        agent_name="Publisher",
        agent_description="Publishing and distribution specialist",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    # Create swarm with interactive dashboard
    swarm = HierarchicalSwarm(
        name="Content-Publishing-Swarm",
        description="Hierarchical swarm for content creation and publishing",
        agents=[content_creator, editor, publisher],
        max_loops=1,
        interactive=True,
        verbose=True,
    )

    # Verify dashboard was created
    assert swarm.dashboard is not None
    assert swarm.interactive is True

    # Execute swarm
    result = swarm.run(
        "Create a comprehensive guide on machine learning best practices"
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
# Streaming tests
##############################################################################


@pytest.fixture
def streaming_swarm():
    """Create a lightweight HierarchicalSwarm for streaming tests."""
    return HierarchicalSwarm(
        name="Streaming-Test-Swarm",
        agents=[
            Agent(
                agent_name="Researcher",
                system_prompt="List 2 short bullets on the topic.",
                model_name="gpt-4.1-mini",
                max_loops=1,
                persistent_memory=False,
                print_on=False,
            ),
            Agent(
                agent_name="Writer",
                system_prompt="Combine into one short paragraph.",
                model_name="gpt-4.1-mini",
                max_loops=1,
                persistent_memory=False,
                print_on=False,
            ),
        ],
        max_loops=1,
        output_type="dict",
        autosave=False,
        director_feedback_on=False,
        planning_enabled=False,
    )


@pytest.mark.asyncio
async def test_arun_stream_plain(streaming_swarm):
    """arun_stream without events yields (agent_name, token) tuples."""
    agent_names = set()
    token_count = 0
    async for agent_name, token in streaming_swarm.arun_stream(
        "solid-state batteries"
    ):
        assert isinstance(agent_name, str)
        assert isinstance(token, str)
        agent_names.add(agent_name)
        token_count += 1

    assert token_count > 5, f"Too few tokens: {token_count}"
    # Director + at least one worker
    assert (
        len(agent_names) >= 2
    ), f"Expected >=2 agents, got {agent_names}"


@pytest.mark.asyncio
async def test_arun_stream_event_types(streaming_swarm):
    """with_events=True emits all required event types with role and loop."""
    event_types = set()
    roles = set()
    end_events = []

    async for evt in streaming_swarm.arun_stream(
        "solid-state batteries", with_events=True
    ):
        assert "type" in evt
        event_types.add(evt["type"])
        if "role" in evt:
            roles.add(evt["role"])
        if evt["type"].endswith("_end"):
            end_events.append(evt)

    # All required event types
    assert "swarm_start" in event_types
    assert "swarm_end" in event_types
    assert "director_start" in event_types
    assert "director_end" in event_types
    assert "worker_start" in event_types
    assert "worker_end" in event_types
    assert "token" in event_types

    # Roles tagged correctly
    assert "director" in roles
    assert "worker" in roles
    assert "swarm" in roles

    # End events (except swarm_end) carry output and loop
    for e in end_events:
        if e["type"] == "swarm_end":
            continue
        assert "loop" in e, f"Missing loop in {e}"
        assert (
            "output" in e and len(str(e["output"])) > 0
        ), f"Missing output in {e}"


@pytest.mark.asyncio
async def test_arun_stream_parallel_interleaving():
    """Parallel workers interleave tokens (>= 3 agent-name flips)."""
    swarm = HierarchicalSwarm(
        name="Parallel-Interleave-Test",
        agents=[
            Agent(
                agent_name="Optimist",
                system_prompt="Write 4-6 upbeat sentences about the topic.",
                model_name="gpt-4.1-mini",
                max_loops=1,
                persistent_memory=False,
                print_on=False,
            ),
            Agent(
                agent_name="Pessimist",
                system_prompt="Write 4-6 cautious sentences about the topic.",
                model_name="gpt-4.1-mini",
                max_loops=1,
                persistent_memory=False,
                print_on=False,
            ),
        ],
        max_loops=1,
        output_type="dict",
        autosave=False,
        director_feedback_on=False,
        planning_enabled=False,
        parallel_execution=True,
    )

    worker_sequence = []
    async for evt in swarm.arun_stream(
        "AI in healthcare", with_events=True
    ):
        if evt["type"] == "token" and evt["role"] == "worker":
            worker_sequence.append(evt["agent"])

    flips = sum(
        1
        for i in range(1, len(worker_sequence))
        if worker_sequence[i] != worker_sequence[i - 1]
    )
    assert (
        len(worker_sequence) > 10
    ), f"Too few worker tokens: {len(worker_sequence)}"
    assert flips >= 3, f"Expected >=3 flips, got {flips}"


@pytest.mark.asyncio
async def test_arun_stream_aggregator_feedback():
    """Aggregator phase streams when director_feedback_on=True."""
    swarm = HierarchicalSwarm(
        name="Aggregator-Test",
        agents=[
            Agent(
                agent_name="Researcher",
                system_prompt="List 2 short bullets on the topic.",
                model_name="gpt-4.1-mini",
                max_loops=1,
                persistent_memory=False,
                print_on=False,
            ),
        ],
        max_loops=1,
        output_type="dict",
        autosave=False,
        director_feedback_on=True,
        planning_enabled=False,
    )

    event_types = set()
    roles = set()
    async for evt in swarm.arun_stream(
        "solid-state batteries", with_events=True
    ):
        event_types.add(evt["type"])
        if "role" in evt:
            roles.add(evt["role"])

    assert "aggregator_start" in event_types
    assert "aggregator_end" in event_types
    assert "aggregator" in roles


@pytest.mark.asyncio
async def test_arun_stream_token_events_have_role_and_loop():
    """Every token event carries role, agent, and loop fields."""
    swarm = HierarchicalSwarm(
        name="Token-Metadata-Test",
        agents=[
            Agent(
                agent_name="Researcher",
                system_prompt="List 2 short bullets on the topic.",
                model_name="gpt-4.1-mini",
                max_loops=1,
                persistent_memory=False,
                print_on=False,
            ),
        ],
        max_loops=1,
        output_type="dict",
        autosave=False,
        director_feedback_on=False,
        planning_enabled=False,
    )

    token_count = 0
    async for evt in swarm.arun_stream(
        "solid-state batteries", with_events=True
    ):
        if evt["type"] == "token":
            token_count += 1
            assert "role" in evt, f"Token missing role: {evt}"
            assert "agent" in evt, f"Token missing agent: {evt}"
            assert "loop" in evt, f"Token missing loop: {evt}"
            assert "token" in evt, f"Token missing token: {evt}"

    assert token_count > 0


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
    """Mimics a nested orchestrator (e.g. another HierarchicalSwarm): it is
    identified by ``name`` (no ``agent_name``) and accepts a
    ``streaming_callback`` kwarg that it calls with the orchestrator's own
    ``(sub_agent_name, chunk, is_final)`` convention.
    """

    def __init__(self, name: str, outputs: list):
        self.name = name
        self.description = f"{name} nested swarm"
        self.outputs = iter(outputs)
        self.calls = 0
        self.output_type = "dict-all-except-first"

    def run(self, task, streaming_callback=None, *args, **kwargs):
        self.calls += 1
        if streaming_callback is not None:
            streaming_callback(self.name, "partial", False)
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return output


class StubFlatOrchestrator:
    """Mimics an orchestrator whose ``run`` has no ``streaming_callback``
    parameter at all (e.g. MixtureOfAgents)."""

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
    assert swarm.max_workers is None
    assert swarm._resolve_max_workers() == max(
        1, int((os.cpu_count() or 1) * 0.75)
    )


def test_max_workers_override_is_used():
    swarm = make_recovery_swarm(
        StubAgent("Director", []),
        [StubAgent("Worker", ["done"])],
        max_workers=2,
    )
    assert swarm.max_workers == 2
    assert swarm._resolve_max_workers() == 2


def test_max_workers_must_be_positive():
    with pytest.raises(ValueError):
        make_recovery_swarm(
            StubAgent("Director", []),
            [StubAgent("Worker", ["done"])],
            max_workers=0,
        )


def test_nested_swarm_worker_is_called_and_streamed():
    sub_team = StubNestedSwarm("SubTeam", ["sub-team result"])
    director = StubAgent("Director", [])
    swarm = make_recovery_swarm(
        director,
        [sub_team],
        parallel_execution=False,
    )

    events = []
    output = swarm.call_single_agent(
        "SubTeam",
        "handle part A",
        streaming_callback=lambda name, chunk, is_final: events.append(
            (name, chunk, is_final)
        ),
    )

    assert output == "sub-team result"
    assert sub_team.calls == 1
    # nested (sub_agent_name, chunk, is_final=False) convention is normalized
    # and re-emitted under the outer worker's own agent_name
    assert ("SubTeam", "[SubTeam] partial", False) in events
    assert ("SubTeam", "", True) in events


def test_orchestrator_without_streaming_param_still_completes():
    flat = StubFlatOrchestrator("FlatOrchestrator", ["moa result"])
    director = StubAgent("Director", [])
    swarm = make_recovery_swarm(
        director,
        [flat],
        parallel_execution=False,
    )

    events = []
    output = swarm.call_single_agent(
        "FlatOrchestrator",
        "handle part B",
        streaming_callback=lambda name, chunk, is_final: events.append(
            (name, chunk, is_final)
        ),
    )

    assert output == "moa result"
    assert flat.calls == 1
    # no incremental chunks possible; a single final completion is emitted
    assert events == [("FlatOrchestrator", "moa result", True)]


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
