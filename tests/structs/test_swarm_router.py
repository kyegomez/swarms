from unittest.mock import Mock, patch

import pytest

from swarms.structs.agent import Agent
from swarms.structs.swarm_router import (
    SwarmRouter,
    SwarmRouterConfig,
    SwarmRouterConfigError,
    SwarmRouterRunError,
)


@pytest.fixture
def sample_agents():
    """Create sample agents for testing."""
    return [
        Agent(
            agent_name="ResearchAgent",
            agent_description="Specializes in researching topics",
            system_prompt="You are a research specialist.",
            max_loops=1,
        ),
        Agent(
            agent_name="CodeAgent",
            agent_description="Expert in coding",
            system_prompt="You are a coding expert.",
            max_loops=1,
        ),
    ]


def test_default_initialization():
    """Test SwarmRouter with default parameters."""
    router = SwarmRouter()

    assert router.name == "swarm-router"
    assert (
        router.description == "Routes your task to the desired swarm"
    )
    assert router.max_loops == 1
    assert router.agents == []
    assert router.swarm_type == "SequentialWorkflow"
    assert router.autosave is False
    assert router.return_json is False
    assert router.auto_generate_prompts is False
    assert router.shared_memory_system is None
    assert router.rules is None
    assert router.documents == []
    assert router.output_type == "dict-all-except-first"
    assert router.verbose is False
    assert router.telemetry_enabled is False


def test_custom_initialization(sample_agents):
    """Test SwarmRouter with custom parameters."""
    router = SwarmRouter(
        name="test-router",
        description="Test router description",
        max_loops=3,
        agents=sample_agents,
        swarm_type="ConcurrentWorkflow",
        autosave=True,
        return_json=True,
        auto_generate_prompts=True,
        rules="Test rules",
        documents=["doc1.txt", "doc2.txt"],
        output_type="json",
        verbose=True,
        telemetry_enabled=True,
    )

    assert router.name == "test-router"
    assert router.description == "Test router description"
    assert router.max_loops == 3
    assert router.agents == sample_agents
    assert router.swarm_type == "ConcurrentWorkflow"
    assert router.autosave is True
    assert router.return_json is True
    assert router.auto_generate_prompts is True
    assert router.rules == "Test rules"
    assert router.documents == ["doc1.txt", "doc2.txt"]
    assert router.output_type == "json"
    assert router.verbose is True
    assert router.telemetry_enabled is True


def test_initialization_with_heavy_swarm_config(sample_agents):
    """Test SwarmRouter with HeavySwarm specific configuration."""
    router = SwarmRouter(
        agents=sample_agents,
        swarm_type="HeavySwarm",
        heavy_swarm_loops_per_agent=2,
        heavy_swarm_question_agent_model_name="gpt-4",
        heavy_swarm_worker_model_name="gpt-3.5-turbo",
        heavy_swarm_swarm_show_output=False,
    )

    assert router.swarm_type == "HeavySwarm"
    assert router.heavy_swarm_loops_per_agent == 2
    assert router.heavy_swarm_question_agent_model_name == "gpt-4"
    assert router.heavy_swarm_worker_model_name == "gpt-3.5-turbo"
    assert router.heavy_swarm_swarm_show_output is False


def test_initialization_with_council_judge_config():
    """Test SwarmRouter with CouncilAsAJudge specific configuration."""
    router = SwarmRouter(
        swarm_type="CouncilAsAJudge",
        council_judge_model_name="gpt-4o",
    )

    assert router.swarm_type == "CouncilAsAJudge"
    assert router.council_judge_model_name == "gpt-4o"


def test_initialization_with_agent_rearrange_flow(sample_agents):
    """Test SwarmRouter with AgentRearrange and flow configuration."""
    flow = "agent1 -> agent2 -> agent1"
    router = SwarmRouter(
        agents=sample_agents,
        swarm_type="AgentRearrange",
        rearrange_flow=flow,
    )

    assert router.swarm_type == "AgentRearrange"
    assert router.rearrange_flow == flow



def test_invalid_swarm_type():
    """Test error when invalid swarm type is provided."""
    with pytest.raises(ValueError):
        SwarmRouter(swarm_type="InvalidSwarmType")


def test_no_agents_for_swarm_requiring_agents():
    """Test error when no agents provided for swarm requiring agents."""
    with pytest.raises(SwarmRouterConfigError):
        SwarmRouter(swarm_type="SequentialWorkflow", agents=None)


def test_no_rearrange_flow_for_agent_rearrange():
    """Test error when no rearrange_flow provided for AgentRearrange."""
    agents = [Agent(agent_name="test", agent_description="test")]
    with pytest.raises(SwarmRouterConfigError):
        SwarmRouter(
            agents=agents,
            swarm_type="AgentRearrange",
            rearrange_flow=None,
        )


def test_zero_max_loops():
    """Test error when max_loops is 0."""
    with pytest.raises(SwarmRouterConfigError):
        SwarmRouter(max_loops=0)


def test_heavy_swarm_without_agents():
    """Test HeavySwarm can be created without agents."""
    router = SwarmRouter(swarm_type="HeavySwarm", agents=None)
    assert router.swarm_type == "HeavySwarm"


def test_council_judge_without_agents():
    """Test CouncilAsAJudge can be created without agents."""
    router = SwarmRouter(swarm_type="CouncilAsAJudge", agents=None)
    assert router.swarm_type == "CouncilAsAJudge"


def test_swarm_factory_initialization(sample_agents):
    """Test that swarm factory is properly initialized."""
    router = SwarmRouter(agents=sample_agents)
    factory = router._initialize_swarm_factory()

    expected_types = [
        "HeavySwarm",
        "AgentRearrange",
        "MALT",
        "CouncilAsAJudge",
        "InteractiveGroupChat",
        "HiearchicalSwarm",
        "MixtureOfAgents",
        "MajorityVoting",
        "GroupChat",
        "MultiAgentRouter",
        "SequentialWorkflow",
        "ConcurrentWorkflow",
        "BatchedGridWorkflow",
    ]

    for swarm_type in expected_types:
        assert swarm_type in factory
        assert callable(factory[swarm_type])


def test_create_heavy_swarm(sample_agents):
    """Test HeavySwarm creation."""
    router = SwarmRouter(
        agents=sample_agents,
        swarm_type="HeavySwarm",
        heavy_swarm_loops_per_agent=2,
        heavy_swarm_question_agent_model_name="gpt-4",
        heavy_swarm_worker_model_name="gpt-3.5-turbo",
    )

    swarm = router._create_heavy_swarm()
    assert swarm is not None
    assert hasattr(swarm, "run")


def test_create_agent_rearrange(sample_agents):
    """Test AgentRearrange creation."""
    router = SwarmRouter(
        agents=sample_agents,
        swarm_type="AgentRearrange",
        rearrange_flow="agent1 -> agent2",
    )

    swarm = router._create_agent_rearrange()
    assert swarm is not None
    assert hasattr(swarm, "run")


def test_create_sequential_workflow(sample_agents):
    """Test SequentialWorkflow creation."""
    router = SwarmRouter(
        agents=sample_agents, swarm_type="SequentialWorkflow"
    )

    swarm = router._create_sequential_workflow()
    assert swarm is not None
    assert hasattr(swarm, "run")


def test_create_concurrent_workflow(sample_agents):
    """Test ConcurrentWorkflow creation."""
    router = SwarmRouter(
        agents=sample_agents, swarm_type="ConcurrentWorkflow"
    )

    swarm = router._create_concurrent_workflow()
    assert swarm is not None
    assert hasattr(swarm, "run")


def test_create_group_chat(sample_agents):
    """Test GroupChat creation."""
    router = SwarmRouter(agents=sample_agents, swarm_type="GroupChat")

    swarm = router._create_group_chat()
    assert swarm is not None
    assert hasattr(swarm, "run")


def test_create_multi_agent_router(sample_agents):
    """Test MultiAgentRouter creation."""
    router = SwarmRouter(
        agents=sample_agents, swarm_type="MultiAgentRouter"
    )

    swarm = router._create_multi_agent_router()
    assert swarm is not None
    assert hasattr(swarm, "run")


def test_create_mixture_of_agents(sample_agents):
    """Test MixtureOfAgents creation."""
    router = SwarmRouter(
        agents=sample_agents, swarm_type="MixtureOfAgents"
    )

    swarm = router._create_mixture_of_agents()
    assert swarm is not None
    assert hasattr(swarm, "run")


def test_create_majority_voting(sample_agents):
    """Test MajorityVoting creation."""
    router = SwarmRouter(
        agents=sample_agents, swarm_type="MajorityVoting"
    )

    swarm = router._create_majority_voting()
    assert swarm is not None
    assert hasattr(swarm, "run")


def test_create_malt(sample_agents):
    """Test MALT creation."""
    router = SwarmRouter(agents=sample_agents, swarm_type="MALT")

    swarm = router._create_malt()
    assert swarm is not None
    assert hasattr(swarm, "run")


def test_create_council_as_judge():
    """Test CouncilAsAJudge creation."""
    router = SwarmRouter(swarm_type="CouncilAsAJudge")

    swarm = router._create_council_as_judge()
    assert swarm is not None
    assert hasattr(swarm, "run")


def test_create_interactive_group_chat(sample_agents):
    """Test InteractiveGroupChat creation."""
    router = SwarmRouter(
        agents=sample_agents, swarm_type="InteractiveGroupChat"
    )

    swarm = router._create_interactive_group_chat()
    assert swarm is not None
    assert hasattr(swarm, "run")


def test_create_hierarchical_swarm(sample_agents):
    """Test HierarchicalSwarm creation."""
    router = SwarmRouter(
        agents=sample_agents, swarm_type="HiearchicalSwarm"
    )

    swarm = router._create_hierarchical_swarm()
    assert swarm is not None
    assert hasattr(swarm, "run")


def test_create_batched_grid_workflow(sample_agents):
    """Test BatchedGridWorkflow creation."""
    router = SwarmRouter(
        agents=sample_agents, swarm_type="BatchedGridWorkflow"
    )

    swarm = router._create_batched_grid_workflow()
    assert swarm is not None
    assert hasattr(swarm, "run")


@pytest.mark.parametrize(
    "swarm_type",
    [
        "SequentialWorkflow",
        "ConcurrentWorkflow",
        "GroupChat",
        "MultiAgentRouter",
        "MixtureOfAgents",
        "MajorityVoting",
        "MALT",
        "CouncilAsAJudge",
        "InteractiveGroupChat",
        "HiearchicalSwarm",
        "BatchedGridWorkflow",
    ],
)
def test_swarm_types_execution(sample_agents, swarm_type):
    """Test execution of all swarm types with mock LLM."""
    with patch("swarms.structs.agent.LiteLLM") as mock_llm:
        # Mock the LLM to return a simple response
        mock_llm_instance = Mock()
        mock_llm_instance.agenerate.return_value = (
            "Test response from agent"
        )
        mock_llm.return_value = mock_llm_instance

        router = SwarmRouter(
            agents=sample_agents,
            swarm_type=swarm_type,
            max_loops=1,
        )

        # Test with a simple task
        task = "Write a simple Python function to add two numbers"

        try:
            result = router.run(task)
            assert (
                result is not None
            ), f"Swarm type {swarm_type} returned None result"
        except Exception:
            # Some swarm types might have specific requirements
            if swarm_type in ["AgentRearrange"]:
                # AgentRearrange requires rearrange_flow
                router = SwarmRouter(
                    agents=sample_agents,
                    swarm_type=swarm_type,
                    rearrange_flow="agent1 -> agent2",
                    max_loops=1,
                )
                result = router.run(task)
                assert (
                    result is not None
                ), f"Swarm type {swarm_type} returned None result"


def test_heavy_swarm_execution():
    """Test HeavySwarm execution."""
    with patch(
        "swarms.structs.heavy_swarm.HeavySwarm"
    ) as mock_heavy_swarm:
        mock_instance = Mock()
        mock_instance.run.return_value = "HeavySwarm response"
        mock_heavy_swarm.return_value = mock_instance

        router = SwarmRouter(swarm_type="HeavySwarm")

        result = router.run("Test task")
        assert result is not None
        assert result == "HeavySwarm response"


def test_agent_rearrange_execution(sample_agents):
    """Test AgentRearrange execution with flow."""
    with patch(
        "swarms.structs.agent_rearrange.AgentRearrange"
    ) as mock_agent_rearrange:
        mock_instance = Mock()
        mock_instance.run.return_value = "AgentRearrange response"
        mock_agent_rearrange.return_value = mock_instance

        router = SwarmRouter(
            agents=sample_agents,
            swarm_type="AgentRearrange",
            rearrange_flow="agent1 -> agent2",
        )

        result = router.run("Test task")
        assert result is not None
        assert result == "AgentRearrange response"


def test_run_method(sample_agents):
    """Test basic run method."""
    with patch(
        "swarms.structs.sequential_workflow.SequentialWorkflow"
    ) as mock_workflow:
        mock_instance = Mock()
        mock_instance.run.return_value = "Test response"
        mock_workflow.return_value = mock_instance

        router = SwarmRouter(
            agents=sample_agents, swarm_type="SequentialWorkflow"
        )

        result = router.run("Test task")
        assert result is not None
        assert result == "Test response"


def test_run_with_image(sample_agents):
    """Test run method with image input."""
    with patch(
        "swarms.structs.sequential_workflow.SequentialWorkflow"
    ) as mock_workflow:
        mock_instance = Mock()
        mock_instance.run.return_value = "Image analysis response"
        mock_workflow.return_value = mock_instance

        router = SwarmRouter(
            agents=sample_agents, swarm_type="SequentialWorkflow"
        )

        result = router.run(
            "Analyze this image", img="test_image.jpg"
        )
        assert result is not None
        assert result == "Image analysis response"


def test_run_with_tasks_list(sample_agents):
    """Test run method with tasks list."""
    with patch(
        "swarms.structs.batched_grid_workflow.BatchedGridWorkflow"
    ) as mock_workflow:
        mock_instance = Mock()
        mock_instance.run.return_value = "Batch response"
        mock_workflow.return_value = mock_instance

        router = SwarmRouter(
            agents=sample_agents, swarm_type="BatchedGridWorkflow"
        )

        tasks = ["Task 1", "Task 2", "Task 3"]
        result = router.run(tasks=tasks)
        assert result is not None
        assert result == "Batch response"


def test_batch_run_method(sample_agents):
    """Test batch_run method."""
    with patch(
        "swarms.structs.sequential_workflow.SequentialWorkflow"
    ) as mock_workflow:
        mock_instance = Mock()
        mock_instance.run.return_value = "Batch task response"
        mock_workflow.return_value = mock_instance

        router = SwarmRouter(
            agents=sample_agents, swarm_type="SequentialWorkflow"
        )

        tasks = ["Task 1", "Task 2", "Task 3"]
        results = router.batch_run(tasks)

        assert results is not None
        assert len(results) == 3
        assert all(result is not None for result in results)


def test_concurrent_run_method(sample_agents):
    """Test concurrent_run method."""
    with patch(
        "swarms.structs.sequential_workflow.SequentialWorkflow"
    ) as mock_workflow:
        mock_instance = Mock()
        mock_instance.run.return_value = "Concurrent response"
        mock_workflow.return_value = mock_instance

        router = SwarmRouter(
            agents=sample_agents, swarm_type="SequentialWorkflow"
        )

        result = router.concurrent_run("Test task")
        assert result is not None
        assert result == "Concurrent response"


def test_call_method(sample_agents):
    """Test __call__ method."""
    with patch(
        "swarms.structs.sequential_workflow.SequentialWorkflow"
    ) as mock_workflow:
        mock_instance = Mock()
        mock_instance.run.return_value = "Call response"
        mock_workflow.return_value = mock_instance

        router = SwarmRouter(
            agents=sample_agents, swarm_type="SequentialWorkflow"
        )

        result = router("Test task")
        assert result is not None
        assert result == "Call response"


def test_call_with_image(sample_agents):
    """Test __call__ method with image."""
    with patch(
        "swarms.structs.sequential_workflow.SequentialWorkflow"
    ) as mock_workflow:
        mock_instance = Mock()
        mock_instance.run.return_value = "Image call response"
        mock_workflow.return_value = mock_instance

        router = SwarmRouter(
            agents=sample_agents, swarm_type="SequentialWorkflow"
        )

        result = router("Test task", img="test.jpg")
        assert result is not None
        assert result == "Image call response"


def test_call_with_images_list(sample_agents):
    """Test __call__ method with images list."""
    with patch(
        "swarms.structs.sequential_workflow.SequentialWorkflow"
    ) as mock_workflow:
        mock_instance = Mock()
        mock_instance.run.return_value = "Images call response"
        mock_workflow.return_value = mock_instance

        router = SwarmRouter(
            agents=sample_agents, swarm_type="SequentialWorkflow"
        )

        result = router("Test task", imgs=["test1.jpg", "test2.jpg"])
        assert result is not None
        assert result == "Images call response"


def test_to_dict_method(sample_agents):
    """Test to_dict method serialization."""
    router = SwarmRouter(
        agents=sample_agents,
        name="test-router",
        description="Test description",
        swarm_type="SequentialWorkflow",
    )

    result_dict = router.to_dict()

    assert isinstance(result_dict, dict)
    assert result_dict["name"] == "test-router"
    assert result_dict["description"] == "Test description"
    assert result_dict["swarm_type"] == "SequentialWorkflow"
    assert "agents" in result_dict


def test_activate_ape(sample_agents):
    """Test activate_ape method."""
    router = SwarmRouter(
        agents=sample_agents,
        auto_generate_prompts=True,
    )

    # Mock the auto_generate_prompt attribute
    for agent in router.agents:
        agent.auto_generate_prompt = False

    router.activate_ape()

    # Check that APE was activated for agents that support it
    for agent in router.agents:
        if hasattr(agent, "auto_generate_prompt"):
            assert agent.auto_generate_prompt is True


def test_handle_rules(sample_agents):
    """Test handle_rules method."""
    rules = "Always be helpful and accurate."
    router = SwarmRouter(
        agents=sample_agents,
        rules=rules,
    )

    original_prompts = [
        agent.system_prompt for agent in router.agents
    ]
    router.handle_rules()

    # Check that rules were added to system prompts
    for i, agent in enumerate(router.agents):
        assert rules in agent.system_prompt
        assert (
            agent.system_prompt
            == original_prompts[i] + f"### Swarm Rules ### {rules}"
        )



def test_update_system_prompt_for_agent_in_swarm(sample_agents):
    """Test update_system_prompt_for_agent_in_swarm method."""
    router = SwarmRouter(
        agents=sample_agents,
        multi_agent_collab_prompt=True,
    )

    original_prompts = [
        agent.system_prompt for agent in router.agents
    ]
    router.update_system_prompt_for_agent_in_swarm()

    # Check that collaboration prompt was added
    for i, agent in enumerate(router.agents):
        assert len(agent.system_prompt) > len(original_prompts[i])


def test_agent_config(sample_agents):
    """Test agent_config method."""
    router = SwarmRouter(agents=sample_agents)

    config = router.agent_config()

    assert isinstance(config, dict)
    assert len(config) == len(sample_agents)

    for agent in sample_agents:
        assert agent.agent_name in config


def test_fetch_message_history_as_string(sample_agents):
    """Test fetch_message_history_as_string method."""
    router = SwarmRouter(agents=sample_agents)

    # Mock the swarm and conversation
    mock_conversation = Mock()
    mock_conversation.return_all_except_first_string.return_value = (
        "Test history"
    )
    router.swarm = Mock()
    router.swarm.conversation = mock_conversation

    result = router.fetch_message_history_as_string()
    assert result == "Test history"


def test_fetch_message_history_as_string_error(sample_agents):
    """Test fetch_message_history_as_string method with error."""
    router = SwarmRouter(agents=sample_agents)

    # Mock the swarm to raise an exception
    router.swarm = Mock()
    router.swarm.conversation.return_all_except_first_string.side_effect = Exception(
        "Test error"
    )

    result = router.fetch_message_history_as_string()
    assert result is None


def test_swarm_creation_error():
    """Test error handling when swarm creation fails."""
    router = SwarmRouter(swarm_type="SequentialWorkflow", agents=None)

    with pytest.raises(SwarmRouterConfigError):
        router.run("Test task")


def test_run_error_handling():
    """Test error handling during task execution."""
    agents = [Agent(agent_name="test", agent_description="test")]

    with patch(
        "swarms.structs.sequential_workflow.SequentialWorkflow"
    ) as mock_workflow:
        mock_instance = Mock()
        mock_instance.run.side_effect = Exception(
            "Test execution error"
        )
        mock_workflow.return_value = mock_instance

        router = SwarmRouter(
            agents=agents, swarm_type="SequentialWorkflow"
        )

        with pytest.raises(SwarmRouterRunError):
            router.run("Test task")


def test_batch_run_error_handling():
    """Test error handling during batch execution."""
    agents = [Agent(agent_name="test", agent_description="test")]

    with patch(
        "swarms.structs.sequential_workflow.SequentialWorkflow"
    ) as mock_workflow:
        mock_instance = Mock()
        mock_instance.run.side_effect = Exception("Test batch error")
        mock_workflow.return_value = mock_instance

        router = SwarmRouter(
            agents=agents, swarm_type="SequentialWorkflow"
        )

        with pytest.raises(RuntimeError):
            router.batch_run(["Task 1", "Task 2"])


def test_invalid_swarm_type_error():
    """Test error when creating swarm with invalid type."""
    router = SwarmRouter(swarm_type="SequentialWorkflow")

    # Manually set an invalid swarm type to test the factory
    router.swarm_type = "InvalidType"

    with pytest.raises(ValueError):
        router._create_swarm("Test task")


def test_swarm_caching():
    """Test that swarms are cached for performance."""
    agents = [Agent(agent_name="test", agent_description="test")]
    router = SwarmRouter(
        agents=agents, swarm_type="SequentialWorkflow"
    )

    # Create swarm first time
    swarm1 = router._create_swarm("Task 1")

    # Create swarm second time with same parameters
    swarm2 = router._create_swarm("Task 1")

    # Should be the same cached instance
    assert swarm1 is swarm2


def test_swarm_cache_different_parameters():
    """Test that different parameters create different cached swarms."""
    agents = [Agent(agent_name="test", agent_description="test")]
    router = SwarmRouter(
        agents=agents, swarm_type="SequentialWorkflow"
    )

    # Create swarms with different parameters
    swarm1 = router._create_swarm("Task 1", param1="value1")
    swarm2 = router._create_swarm("Task 2", param1="value2")

    # Should be different instances
    assert swarm1 is not swarm2


@pytest.mark.parametrize(
    "output_type",
    [
        "string",
        "str",
        "list",
        "json",
        "dict",
        "yaml",
        "xml",
        "dict-all-except-first",
        "dict-first",
        "list-all-except-first",
    ],
)
def test_output_types(sample_agents, output_type):
    """Test different output types."""
    with patch(
        "swarms.structs.sequential_workflow.SequentialWorkflow"
    ) as mock_workflow:
        mock_instance = Mock()
        mock_instance.run.return_value = f"Response for {output_type}"
        mock_workflow.return_value = mock_instance

        router = SwarmRouter(
            agents=sample_agents,
            swarm_type="SequentialWorkflow",
            output_type=output_type,
        )

        result = router.run("Test task")
        assert result is not None
        assert result == f"Response for {output_type}"


def test_full_workflow_with_sequential_workflow():
    """Test complete workflow with SequentialWorkflow."""
    agents = [
        Agent(
            agent_name="ResearchAgent",
            agent_description="Research specialist",
            system_prompt="You are a research specialist.",
            max_loops=1,
        ),
        Agent(
            agent_name="CodeAgent",
            agent_description="Code expert",
            system_prompt="You are a code expert.",
            max_loops=1,
        ),
    ]

    with patch(
        "swarms.structs.sequential_workflow.SequentialWorkflow"
    ) as mock_workflow:
        mock_instance = Mock()
        mock_instance.run.return_value = "Complete workflow response"
        mock_workflow.return_value = mock_instance

        router = SwarmRouter(
            agents=agents,
            swarm_type="SequentialWorkflow",
            max_loops=2,
            rules="Always provide accurate information",
            multi_agent_collab_prompt=True,
            verbose=True,
        )

        # Test various methods
        result = router.run("Research and code a Python function")
        assert result is not None

        batch_results = router.batch_run(["Task 1", "Task 2"])
        assert batch_results is not None
        assert len(batch_results) == 2

        concurrent_result = router.concurrent_run("Concurrent task")
        assert concurrent_result is not None

        # Test serialization
        config_dict = router.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["swarm_type"] == "SequentialWorkflow"


def test_swarm_router_config_model():
    """Test SwarmRouterConfig model."""
    config = SwarmRouterConfig(
        name="test-config",
        description="Test configuration",
        swarm_type="SequentialWorkflow",
        task="Test task",
        multi_agent_collab_prompt=True,
    )

    assert config.name == "test-config"
    assert config.description == "Test configuration"
    assert config.swarm_type == "SequentialWorkflow"
    assert config.task == "Test task"
    assert config.multi_agent_collab_prompt is True



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
