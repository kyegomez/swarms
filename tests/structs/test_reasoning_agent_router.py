"""Pytest tests for ReasoningAgentRouter and related exceptions."""

import pytest
from unittest.mock import MagicMock, patch

from swarms.agents.reasoning_agent_router import (
    ReasoningAgentExecutorError,
    ReasoningAgentInitializationError,
    ReasoningAgentRouter,
)


# ---------------------------------------------------------------------------
# __init__ and reliability_check (via init)
# ---------------------------------------------------------------------------


def test_router_init_default():
    """Test ReasoningAgentRouter initialization with defaults."""
    router = ReasoningAgentRouter()
    assert router.agent_name == "reasoning_agent"
    assert (
        router.description
        == "A reasoning agent that can answer questions and help with tasks."
    )
    assert router.model_name == "gpt-4o-mini"
    assert (
        router.system_prompt
        == "You are a helpful assistant that can answer questions and help with tasks."
    )
    assert router.max_loops == 1
    assert router.swarm_type == "reasoning-duo"
    assert router.num_samples == 1
    assert router.num_knowledge_items == 6
    assert router.memory_capacity == 6
    assert router.eval is False
    assert router.random_models_on is False
    assert router.majority_voting_prompt is None
    assert router.reasoning_model_name == "gpt-4o"
    assert router.agent_factories is not None


def test_router_init_custom():
    """Test ReasoningAgentRouter with custom parameters."""
    router = ReasoningAgentRouter(
        agent_name="custom_agent",
        description="Custom desc",
        model_name="gpt-4",
        system_prompt="Custom prompt",
        max_loops=5,
        swarm_type="self-consistency",
        num_samples=3,
        num_knowledge_items=10,
        memory_capacity=20,
        eval=True,
        random_models_on=True,
        majority_voting_prompt="Vote prompt",
        reasoning_model_name="claude-3",
    )
    assert router.agent_name == "custom_agent"
    assert router.description == "Custom desc"
    assert router.model_name == "gpt-4"
    assert router.max_loops == 5
    assert router.swarm_type == "self-consistency"
    assert router.num_samples == 3
    assert router.num_knowledge_items == 10
    assert router.memory_capacity == 20
    assert router.eval is True
    assert router.random_models_on is True
    assert router.majority_voting_prompt == "Vote prompt"
    assert router.reasoning_model_name == "claude-3"


def test_reliability_check_max_loops_zero():
    """reliability_check raises when max_loops is 0."""
    with pytest.raises(ReasoningAgentInitializationError) as exc_info:
        ReasoningAgentRouter(max_loops=0)
    assert "Max loops must be greater than 0" in str(exc_info.value)


def test_reliability_check_model_name_empty():
    """reliability_check raises when model_name is empty."""
    with pytest.raises(ReasoningAgentInitializationError) as exc_info:
        ReasoningAgentRouter(model_name="")
    assert "Model name must be provided" in str(exc_info.value)


def test_reliability_check_model_name_none():
    """reliability_check raises when model_name is None."""
    with pytest.raises(ReasoningAgentInitializationError) as exc_info:
        ReasoningAgentRouter(model_name=None)
    assert "Model name must be provided" in str(exc_info.value)


def test_reliability_check_swarm_type_empty():
    """reliability_check raises when swarm_type is empty."""
    with pytest.raises(ReasoningAgentInitializationError) as exc_info:
        ReasoningAgentRouter(swarm_type="")
    assert "Swarm type must be provided" in str(exc_info.value)


def test_reliability_check_swarm_type_none():
    """reliability_check raises when swarm_type is None."""
    with pytest.raises(ReasoningAgentInitializationError) as exc_info:
        ReasoningAgentRouter(swarm_type=None)
    assert "Swarm type must be provided" in str(exc_info.value)


# ---------------------------------------------------------------------------
# _initialize_agent_factories
# ---------------------------------------------------------------------------


def test_initialize_agent_factories_keys():
    """_initialize_agent_factories returns dict with expected agent type keys."""
    router = ReasoningAgentRouter()
    factories = router.agent_factories
    expected_keys = {
        "reasoning-duo",
        "reasoning-agent",
        "self-consistency",
        "consistency-agent",
        "ire",
        "ire-agent",
        "AgentJudge",
        "ReflexionAgent",
        "GKPAgent",
    }
    assert set(factories.keys()) == expected_keys
    for v in factories.values():
        assert callable(v)


# ---------------------------------------------------------------------------
# Factory methods: _create_reasoning_duo, _create_consistency_agent, etc.
# ---------------------------------------------------------------------------


def test_create_reasoning_duo():
    """_create_reasoning_duo returns a ReasoningDuo instance with correct config."""
    from swarms.agents.reasoning_duo import ReasoningDuo

    router = ReasoningAgentRouter(
        swarm_type="reasoning-duo",
        agent_name="rd",
        model_name="gpt-4o-mini",
        system_prompt="SP",
        max_loops=2,
        reasoning_model_name="gpt-4o",
    )
    agent = router._create_reasoning_duo()
    assert isinstance(agent, ReasoningDuo)
    assert agent.agent_name == "rd"
    assert agent.agent_description == router.description
    assert agent.model_name == [router.model_name, router.model_name]
    assert agent.system_prompt == "SP"
    assert agent.max_loops == 2
    assert agent.reasoning_model_name == "gpt-4o"


def test_create_consistency_agent():
    """_create_consistency_agent returns SelfConsistencyAgent with correct config."""
    from swarms.agents.consistency_agent import SelfConsistencyAgent

    router = ReasoningAgentRouter(
        swarm_type="self-consistency",
        agent_name="sc",
        description="Desc",
        model_name="gpt-4o-mini",
        system_prompt="SP",
        max_loops=3,
        num_samples=5,
        eval=True,
        random_models_on=True,
        majority_voting_prompt="Vote",
    )
    agent = router._create_consistency_agent()
    assert isinstance(agent, SelfConsistencyAgent)
    assert agent.name == "sc"
    assert agent.description == "Desc"
    assert agent.model_name == "gpt-4o-mini"
    assert agent.system_prompt == "SP"
    assert agent.max_loops == 3
    assert agent.num_samples == 5
    assert agent.eval is True
    assert agent.random_models_on is True
    assert agent.majority_voting_prompt == "Vote"


def test_create_ire_agent():
    """_create_ire_agent returns IREAgent with correct config."""
    from swarms.agents.i_agent import (
        IterativeReflectiveExpansion as IREAgent,
    )

    router = ReasoningAgentRouter(
        swarm_type="ire",
        agent_name="ire",
        description="IRE desc",
        model_name="gpt-4o-mini",
        system_prompt="SP",
        num_samples=4,
    )
    agent = router._create_ire_agent()
    assert isinstance(agent, IREAgent)
    assert agent.agent_name == "ire"
    assert agent.description == "IRE desc"
    assert agent.model_name == "gpt-4o-mini"
    assert agent.system_prompt == "SP"
    assert agent.max_loops == 4  # uses num_samples for IRE


def test_create_agent_judge():
    """_create_agent_judge returns AgentJudge with correct config."""
    from swarms.agents.agent_judge import AgentJudge

    router = ReasoningAgentRouter(
        swarm_type="AgentJudge",
        agent_name="judge",
        model_name="gpt-4o-mini",
        system_prompt="SP",
        max_loops=2,
    )
    agent = router._create_agent_judge()
    assert isinstance(agent, AgentJudge)
    assert agent.agent_name == "judge"
    assert agent.model_name == "gpt-4o-mini"
    assert agent.system_prompt == "SP"
    assert agent.max_loops == 2


def test_create_reflexion_agent():
    """_create_reflexion_agent returns ReflexionAgent with correct config."""
    from swarms.agents.flexion_agent import ReflexionAgent

    router = ReasoningAgentRouter(
        swarm_type="ReflexionAgent",
        agent_name="reflex",
        model_name="gpt-4o-mini",
        system_prompt="SP",
        max_loops=2,
        memory_capacity=10,
    )
    agent = router._create_reflexion_agent()
    assert isinstance(agent, ReflexionAgent)
    assert agent.agent_name == "reflex"
    assert agent.model_name == "gpt-4o-mini"
    assert agent.system_prompt == "SP"
    assert agent.max_loops == 2
    assert agent.memory_capacity == 10


def test_create_gkp_agent():
    """_create_gkp_agent returns GKPAgent with correct config."""
    from swarms.agents.gkp_agent import GKPAgent

    router = ReasoningAgentRouter(
        swarm_type="GKPAgent",
        agent_name="gkp",
        model_name="gpt-4o-mini",
        num_knowledge_items=8,
    )
    agent = router._create_gkp_agent()
    assert isinstance(agent, GKPAgent)
    assert agent.agent_name == "gkp"
    assert agent.model_name == "gpt-4o-mini"
    assert agent.num_knowledge_items == 8


# ---------------------------------------------------------------------------
# select_swarm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "swarm_type",
    [
        "reasoning-duo",
        "reasoning-agent",
        "self-consistency",
        "consistency-agent",
        "ire",
        "ire-agent",
        "ReflexionAgent",
        "GKPAgent",
        "AgentJudge",
    ],
)
def test_select_swarm_valid_types(swarm_type):
    """select_swarm returns a non-None agent for each valid swarm_type."""
    router = ReasoningAgentRouter(swarm_type=swarm_type)
    swarm = router.select_swarm()
    assert swarm is not None


def test_select_swarm_invalid_type():
    """select_swarm raises ReasoningAgentInitializationError for unknown type."""
    router = ReasoningAgentRouter(swarm_type="invalid_type")
    with pytest.raises(ReasoningAgentInitializationError) as exc_info:
        router.select_swarm()
    assert "Invalid swarm type" in str(exc_info.value)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def test_run_calls_swarm_run_with_task():
    """run selects swarm and calls run(task=task) for non-ReflexionAgent."""
    router = ReasoningAgentRouter(swarm_type="reasoning-duo")
    mock_swarm = MagicMock()
    mock_swarm.run.return_value = "result"

    with patch.object(
        router, "select_swarm", return_value=mock_swarm
    ):
        out = router.run("What is 2+2?")
    mock_swarm.run.assert_called_once_with(task="What is 2+2?")
    assert out == "result"


def test_run_calls_reflexion_with_tasks_list():
    """run calls swarm.run(tasks=[task]) for ReflexionAgent."""
    router = ReasoningAgentRouter(swarm_type="ReflexionAgent")
    mock_swarm = MagicMock()
    mock_swarm.run.return_value = "reflex_result"

    with patch.object(
        router, "select_swarm", return_value=mock_swarm
    ):
        out = router.run("Do something")
    mock_swarm.run.assert_called_once_with(tasks=["Do something"])
    assert out == "reflex_result"


def test_run_passes_args_kwargs():
    """run forwards *args and **kwargs to swarm.run."""
    router = ReasoningAgentRouter(swarm_type="ire")
    mock_swarm = MagicMock()
    mock_swarm.run.return_value = "ok"

    with patch.object(
        router, "select_swarm", return_value=mock_swarm
    ):
        router.run("task", "extra_arg", key="value")
    mock_swarm.run.assert_called_once_with(task="task", key="value")


def test_run_raises_executor_error_on_swarm_failure():
    """run raises ReasoningAgentExecutorError when swarm.run raises."""
    router = ReasoningAgentRouter(swarm_type="reasoning-duo")
    mock_swarm = MagicMock()
    mock_swarm.run.side_effect = ValueError("agent failed")

    with patch.object(
        router, "select_swarm", return_value=mock_swarm
    ):
        with pytest.raises(ReasoningAgentExecutorError) as exc_info:
            router.run("task")
    assert "ReasoningAgentRouter Error" in str(exc_info.value)
    assert "agent failed" in str(exc_info.value)


def test_run_raises_executor_error_when_select_swarm_fails():
    """run raises ReasoningAgentExecutorError when select_swarm raises."""
    router = ReasoningAgentRouter(swarm_type="reasoning-duo")
    with patch.object(
        router,
        "select_swarm",
        side_effect=RuntimeError("init failed"),
    ):
        with pytest.raises(ReasoningAgentExecutorError) as exc_info:
            router.run("task")
    assert "ReasoningAgentRouter Error" in str(exc_info.value)


# ---------------------------------------------------------------------------
# batched_run
# ---------------------------------------------------------------------------


def test_batched_run_calls_run_per_task():
    """batched_run calls run once per task and returns list of results."""
    router = ReasoningAgentRouter(swarm_type="reasoning-duo")
    tasks = ["task1", "task2", "task3"]
    with patch.object(
        router, "run", side_effect=["r1", "r2", "r3"]
    ) as mock_run:
        results = router.batched_run(tasks)
    assert mock_run.call_count == 3
    assert mock_run.call_args_list[0][0][0] == "task1"
    assert mock_run.call_args_list[1][0][0] == "task2"
    assert mock_run.call_args_list[2][0][0] == "task3"
    assert results == ["r1", "r2", "r3"]


def test_batched_run_empty_list():
    """batched_run with empty tasks returns empty list."""
    router = ReasoningAgentRouter(swarm_type="reasoning-duo")
    with patch.object(router, "run") as mock_run:
        results = router.batched_run([])
    mock_run.assert_not_called()
    assert results == []


def test_batched_run_passes_args_kwargs():
    """batched_run passes *args and **kwargs to each run call."""
    router = ReasoningAgentRouter(swarm_type="reasoning-duo")
    with patch.object(router, "run", return_value="x") as mock_run:
        router.batched_run(["a", "b"], "arg", kw="val")
    assert mock_run.call_count == 2
    mock_run.assert_any_call("a", "arg", kw="val")
    mock_run.assert_any_call("b", "arg", kw="val")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


def test_reasoning_agent_initialization_error_subclass():
    """ReasoningAgentInitializationError is a subclass of Exception."""
    assert issubclass(ReasoningAgentInitializationError, Exception)


def test_reasoning_agent_executor_error_subclass():
    """ReasoningAgentExecutorError is a subclass of Exception."""
    assert issubclass(ReasoningAgentExecutorError, Exception)
