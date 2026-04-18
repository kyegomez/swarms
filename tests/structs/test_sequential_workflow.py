import os
from unittest.mock import patch

import pytest

from swarms import Agent, SequentialWorkflow
from swarms.structs.sequential_workflow import DRIFT_DETECTION_PROMPT
from swarms.utils.workspace_utils import get_workspace_dir


def test_sequential_workflow_initialization_with_agents():
    """Test SequentialWorkflow initialization with agents"""
    agent1 = Agent(
        agent_name="Agent-1",
        agent_description="First test agent",
        model_name="gpt-4o",
        max_loops=1,
    )
    agent2 = Agent(
        agent_name="Agent-2",
        agent_description="Second test agent",
        model_name="gpt-4o",
        max_loops=1,
    )

    workflow = SequentialWorkflow(
        name="Test-Workflow",
        description="Test workflow with multiple agents",
        agents=[agent1, agent2],
        max_loops=1,
    )

    assert isinstance(workflow, SequentialWorkflow)
    assert workflow.name == "Test-Workflow"
    assert (
        workflow.description == "Test workflow with multiple agents"
    )
    assert len(workflow.agents) == 2
    assert workflow.agents[0] == agent1
    assert workflow.agents[1] == agent2
    assert workflow.max_loops == 1


def test_sequential_workflow_multi_agent_execution():
    """Test SequentialWorkflow execution with multiple agents"""
    agent1 = Agent(
        agent_name="Research-Agent",
        agent_description="Agent for research tasks",
        model_name="gpt-4o",
        max_loops=1,
    )
    agent2 = Agent(
        agent_name="Analysis-Agent",
        agent_description="Agent for analyzing research results",
        model_name="gpt-4o",
        max_loops=1,
    )
    agent3 = Agent(
        agent_name="Summary-Agent",
        agent_description="Agent for summarizing findings",
        model_name="gpt-4o",
        max_loops=1,
    )

    workflow = SequentialWorkflow(
        name="Multi-Agent-Research-Workflow",
        description="Workflow for comprehensive research, analysis, and summarization",
        agents=[agent1, agent2, agent3],
        max_loops=1,
    )

    # Test that the workflow executes successfully
    result = workflow.run(
        "Analyze the impact of renewable energy on climate change"
    )
    assert result is not None
    # SequentialWorkflow may return different types based on output_type, just ensure it's not None


def test_sequential_workflow_batched_execution():
    """Test batched execution of SequentialWorkflow"""
    agent1 = Agent(
        agent_name="Data-Collector",
        agent_description="Agent for collecting data",
        model_name="gpt-4o",
        max_loops=1,
    )
    agent2 = Agent(
        agent_name="Data-Processor",
        agent_description="Agent for processing collected data",
        model_name="gpt-4o",
        max_loops=1,
    )

    workflow = SequentialWorkflow(
        name="Batched-Processing-Workflow",
        agents=[agent1, agent2],
        max_loops=1,
    )

    # Test batched execution
    tasks = [
        "Analyze solar energy trends",
        "Evaluate wind power efficiency",
        "Compare renewable energy sources",
    ]
    results = workflow.run_batched(tasks)
    assert results is not None
    # run_batched returns a list of results
    assert isinstance(results, list)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_sequential_workflow_async_execution():
    """Test async execution of SequentialWorkflow"""
    agent1 = Agent(
        agent_name="Async-Research-Agent",
        agent_description="Agent for async research tasks",
        model_name="gpt-4o",
        max_loops=1,
    )
    agent2 = Agent(
        agent_name="Async-Analysis-Agent",
        agent_description="Agent for async analysis",
        model_name="gpt-4o",
        max_loops=1,
    )

    workflow = SequentialWorkflow(
        name="Async-Workflow",
        agents=[agent1, agent2],
        max_loops=1,
    )

    # Test async execution
    result = await workflow.run_async("Analyze AI trends in 2024")
    assert result is not None


@pytest.mark.asyncio
async def test_sequential_workflow_concurrent_execution():
    """Test concurrent execution of SequentialWorkflow"""
    agent1 = Agent(
        agent_name="Concurrent-Research-Agent",
        agent_description="Agent for concurrent research",
        model_name="gpt-4o",
        max_loops=1,
    )
    agent2 = Agent(
        agent_name="Concurrent-Analysis-Agent",
        agent_description="Agent for concurrent analysis",
        model_name="gpt-4o",
        max_loops=1,
    )
    agent3 = Agent(
        agent_name="Concurrent-Summary-Agent",
        agent_description="Agent for concurrent summarization",
        model_name="gpt-4o",
        max_loops=1,
    )

    workflow = SequentialWorkflow(
        name="Concurrent-Workflow",
        agents=[agent1, agent2, agent3],
        max_loops=1,
    )

    # Test concurrent execution
    tasks = [
        "Research quantum computing advances",
        "Analyze blockchain technology trends",
        "Evaluate machine learning applications",
    ]
    results = await workflow.run_concurrent(tasks)
    assert results is not None
    # run_concurrent returns a list of results
    assert isinstance(results, list)
    assert len(results) == 3


def test_sequential_workflow_with_multi_agent_collaboration():
    """Test SequentialWorkflow with multi-agent collaboration prompts"""
    agent1 = Agent(
        agent_name="Market-Research-Agent",
        agent_description="Agent for market research",
        model_name="gpt-4o",
        max_loops=1,
    )
    agent2 = Agent(
        agent_name="Competitive-Analysis-Agent",
        agent_description="Agent for competitive analysis",
        model_name="gpt-4o",
        max_loops=1,
    )
    agent3 = Agent(
        agent_name="Strategy-Development-Agent",
        agent_description="Agent for developing business strategies",
        model_name="gpt-4o",
        max_loops=1,
    )

    workflow = SequentialWorkflow(
        name="Business-Strategy-Workflow",
        description="Comprehensive business strategy development workflow",
        agents=[agent1, agent2, agent3],
        max_loops=1,
        multi_agent_collab_prompt=True,
    )

    # Test that collaboration prompt is added
    assert agent1.system_prompt is not None
    assert agent2.system_prompt is not None
    assert agent3.system_prompt is not None

    # Test execution
    result = workflow.run(
        "Develop a business strategy for entering the AI market"
    )
    assert result is not None


def test_sequential_workflow_error_handling():
    """Test SequentialWorkflow error handling"""
    # Test with invalid agents list
    with pytest.raises(
        ValueError, match="Agents list cannot be None or empty"
    ):
        SequentialWorkflow(agents=None)

    with pytest.raises(
        ValueError, match="Agents list cannot be None or empty"
    ):
        SequentialWorkflow(agents=[])

    # Test with zero max_loops
    with pytest.raises(ValueError, match="max_loops cannot be 0"):
        agent1 = Agent(
            agent_name="Test-Agent",
            agent_description="Test agent",
            model_name="gpt-4o",
            max_loops=1,
        )
        SequentialWorkflow(agents=[agent1], max_loops=0)


def test_sequential_workflow_agent_names_extraction():
    """Test that SequentialWorkflow properly extracts agent names for flow"""
    agent1 = Agent(
        agent_name="Alpha-Agent",
        agent_description="First agent",
        model_name="gpt-4o",
        max_loops=1,
    )
    agent2 = Agent(
        agent_name="Beta-Agent",
        agent_description="Second agent",
        model_name="gpt-4o",
        max_loops=1,
    )
    agent3 = Agent(
        agent_name="Gamma-Agent",
        agent_description="Third agent",
        model_name="gpt-4o",
        max_loops=1,
    )

    workflow = SequentialWorkflow(
        name="Test-Flow-Workflow",
        agents=[agent1, agent2, agent3],
        max_loops=1,
    )

    # Test flow string generation
    expected_flow = "Alpha-Agent -> Beta-Agent -> Gamma-Agent"
    assert workflow.flow == expected_flow


def test_sequential_workflow_team_awareness():
    """Test SequentialWorkflow with team awareness enabled"""
    agent1 = Agent(
        agent_name="Team-Member-1",
        agent_description="First team member",
        model_name="gpt-4o",
        max_loops=1,
    )
    agent2 = Agent(
        agent_name="Team-Member-2",
        agent_description="Second team member",
        model_name="gpt-4o",
        max_loops=1,
    )

    workflow = SequentialWorkflow(
        name="Team-Aware-Workflow",
        description="Workflow with team awareness",
        agents=[agent1, agent2],
        max_loops=1,
        team_awareness=True,
    )

    # Test that workflow initializes successfully with team awareness
    assert workflow.team_awareness is True
    assert len(workflow.agents) == 2


def test_sequential_workflow_autosave_creates_workspace_dir(
    monkeypatch, tmp_path
):
    """Test that SequentialWorkflow with autosave=True creates a workspace directory."""
    get_workspace_dir.cache_clear()
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))

    agent1 = Agent(
        agent_name="Autosave-Agent-1",
        agent_description="Agent for autosave test",
        model_name="gpt-5.4",
        max_loops=1,
    )
    agent2 = Agent(
        agent_name="Autosave-Agent-2",
        agent_description="Agent for autosave test",
        model_name="gpt-5.4",
        max_loops=1,
    )

    workflow = SequentialWorkflow(
        name="Autosave-Test-Workflow",
        agents=[agent1, agent2],
        max_loops=1,
        autosave=True,
    )

    assert workflow.autosave is True
    assert workflow.swarm_workspace_dir is not None
    assert os.path.isdir(workflow.swarm_workspace_dir)
    assert "SequentialWorkflow" in workflow.swarm_workspace_dir
    assert "Autosave-Test-Workflow" in workflow.swarm_workspace_dir

    get_workspace_dir.cache_clear()


def test_sequential_workflow_autosave_saves_conversation_after_run(
    monkeypatch, tmp_path
):
    """Test that SequentialWorkflow saves conversation_history.json after run when autosave=True."""
    get_workspace_dir.cache_clear()
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))

    agent1 = Agent(
        agent_name="Autosave-Run-Agent-1",
        agent_description="Agent for autosave run test",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )
    agent2 = Agent(
        agent_name="Autosave-Run-Agent-2",
        agent_description="Agent for autosave run test",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    workflow = SequentialWorkflow(
        name="Autosave-Run-Workflow",
        agents=[agent1, agent2],
        max_loops=1,
        autosave=True,
        verbose=False,
    )

    result = workflow.run("Say hello in one short sentence.")
    assert result is not None

    conversation_path = os.path.join(
        workflow.swarm_workspace_dir, "conversation_history.json"
    )
    assert os.path.isfile(
        conversation_path
    ), f"Expected conversation_history.json at {conversation_path}"

    get_workspace_dir.cache_clear()


# ---------------------------------------------------------------------------
# Drift detection integration
# ---------------------------------------------------------------------------


def _make_workflow(drift_detection=False, **kwargs):
    a1 = Agent(agent_name="A1", model_name="gpt-4o", max_loops=1)
    a2 = Agent(agent_name="A2", model_name="gpt-4o", max_loops=1)
    return SequentialWorkflow(
        agents=[a1, a2],
        max_loops=1,
        autosave=False,
        drift_detection=drift_detection,
        **kwargs,
    )


def test_drift_detection_prompt_exists():
    assert isinstance(DRIFT_DETECTION_PROMPT, str)
    assert len(DRIFT_DETECTION_PROMPT) > 0


def test_workflow_drift_agent_none_when_disabled():
    wf = _make_workflow(drift_detection=False)
    assert wf.drift_agent is None


def test_workflow_drift_agent_is_agent_when_enabled():
    wf = _make_workflow(drift_detection=True)
    assert isinstance(wf.drift_agent, Agent)
    assert wf.drift_agent.agent_name == "DriftDetector"


def test_workflow_drift_agent_uses_custom_model():
    wf = _make_workflow(
        drift_detection=True, drift_model="gpt-4o-mini"
    )
    assert wf.drift_agent.model_name == "gpt-4o-mini"


def test_workflow_drift_threshold_stored():
    wf = _make_workflow(drift_detection=True, drift_threshold=0.9)
    assert wf.drift_threshold == 0.9


def test_workflow_run_without_drift_returns_raw():
    wf = _make_workflow(drift_detection=False)
    fake_output = "pipeline result"
    with patch.object(
        wf.agent_rearrange, "run", return_value=fake_output
    ):
        result = wf.run("do something")
    assert result == fake_output


def _tool_call_response(score: float) -> str:
    """Return a mock drift agent response in the actual tool call format."""
    import json

    return str(
        [
            {
                "index": 1,
                "caller": {"type": "direct"},
                "function": {
                    "arguments": json.dumps({"score": score}),
                    "name": "score_drift",
                },
                "id": "toolu_mock",
                "type": "function",
            }
        ]
    )


def test_workflow_run_with_drift_above_threshold_no_warning(caplog):
    wf = _make_workflow(drift_detection=True, drift_threshold=0.75)
    with patch.object(
        wf.agent_rearrange, "run", return_value="good output"
    ), patch.object(
        wf.drift_agent, "run", return_value=_tool_call_response(0.95)
    ):
        result = wf.run("task")
    assert result == "good output"
    assert "Drift detected" not in caplog.text


def test_workflow_run_with_drift_below_threshold_warns():
    from unittest.mock import MagicMock
    import swarms.structs.sequential_workflow as sw_module

    wf = _make_workflow(drift_detection=True, drift_threshold=0.75)
    # Return low score first (triggers warning + rerun), then passing score to terminate
    drift_responses = iter(
        [_tool_call_response(0.3), _tool_call_response(0.9)]
    )
    mock_logger = MagicMock()
    with patch.object(
        wf.agent_rearrange, "run", return_value="off-topic output"
    ), patch.object(
        wf.drift_agent,
        "run",
        side_effect=lambda *a, **kw: next(drift_responses),
    ), patch.object(
        sw_module, "logger", mock_logger
    ):
        result = wf.run("task")
    assert result == "off-topic output"
    warning_calls = [
        str(c) for c in mock_logger.warning.call_args_list
    ]
    assert any("Drift detected" in c for c in warning_calls)


def test_workflow_run_drift_agent_failure_does_not_raise():
    wf = _make_workflow(drift_detection=True)
    with patch.object(
        wf.agent_rearrange, "run", return_value="output"
    ), patch.object(
        wf.drift_agent, "run", side_effect=RuntimeError("LLM error")
    ):
        result = wf.run("task")
    assert result == "output"


def test_workflow_run_drift_unparseable_response_does_not_raise():
    wf = _make_workflow(drift_detection=True)
    with patch.object(
        wf.agent_rearrange, "run", return_value="output"
    ), patch.object(
        wf.drift_agent, "run", return_value="not-a-tool-call"
    ):
        result = wf.run("task")
    assert result == "output"


def test_workflow_run_drift_retries_until_threshold_met():
    """Pipeline reruns until drift score meets threshold."""
    wf = _make_workflow(drift_detection=True, drift_threshold=0.75)
    pipeline_outputs = ["bad output", "bad output", "good output"]
    drift_scores = [0.3, 0.5, 0.9]

    pipeline_call_count = 0

    def pipeline_side_effect(**kwargs):
        nonlocal pipeline_call_count
        result = pipeline_outputs[
            min(pipeline_call_count, len(pipeline_outputs) - 1)
        ]
        pipeline_call_count += 1
        return result

    drift_call_count = 0

    def drift_side_effect(prompt):
        nonlocal drift_call_count
        score = drift_scores[
            min(drift_call_count, len(drift_scores) - 1)
        ]
        drift_call_count += 1
        return _tool_call_response(score)

    with patch.object(
        wf.agent_rearrange, "run", side_effect=pipeline_side_effect
    ), patch.object(
        wf.drift_agent, "run", side_effect=drift_side_effect
    ):
        result = wf.run("task")

    assert result == "good output"
    assert pipeline_call_count == 3
    assert drift_call_count == 3
