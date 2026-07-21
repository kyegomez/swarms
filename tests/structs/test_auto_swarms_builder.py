import json

import pytest
from dotenv import load_dotenv

from swarms.structs.agent import Agent
from swarms.structs.auto_swarm_builder import (
    AgentSpec,
    AutoSwarmBuilder,
)
from swarms.structs.ma_utils import set_random_models_for_agents

load_dotenv()


def print_separator():
    """Print a separator line for test output formatting."""
    print("\n" + "=" * 50)


def test_initialization():
    """Test basic initialization of AutoSwarmBuilder"""
    print_separator()
    print("Testing AutoSwarmBuilder Initialization")
    try:
        swarm = AutoSwarmBuilder(
            name="TestSwarm",
            description="A test swarm for validation",
            verbose=True,
            max_loops=2,
        )

        print("✓ Created swarm with configuration:")
        print(f"  - Name: {swarm.name}")
        print(f"  - Description: {swarm.description}")
        print(f"  - Max loops: {swarm.max_loops}")
        print(f"  - Verbose: {swarm.verbose}")
        print("✓ Initialization test passed")
        return swarm
    except Exception as e:
        print(f"✗ Initialization test failed: {str(e)}")
        raise


def test_agent_building():
    """Test building individual agents from specs"""
    print_separator()
    print("Testing Agent Building")
    try:
        swarm = AutoSwarmBuilder()
        specs = {
            "agents": [
                {
                    "agent_name": "TestAgent",
                    "description": "A test agent",
                    "system_prompt": "You are a test agent",
                    "max_loops": 1,
                }
            ]
        }
        agents = swarm.create_agents_from_specs(specs)
        agent = agents[0]

        print("✓ Built agent with configuration:")
        print(f"  - Name: {agent.agent_name}")
        print(f"  - Description: {agent.agent_description}")
        print(f"  - Max loops: {agent.max_loops}")
        print("✓ Agent building test passed")
        return agent
    except Exception as e:
        print(f"✗ Agent building test failed: {str(e)}")
        raise


def test_agent_creation():
    """Test creating multiple agents for a task"""
    print_separator()
    print("Testing Agent Creation from Task")
    try:
        swarm = AutoSwarmBuilder(
            name="ResearchSwarm",
            description="A swarm for research tasks",
        )
        task = "Research the latest developments in quantum computing"
        # create_agents returns an already-parsed dict: {"agents": [...]}
        agent_specs = swarm.create_agents(task)
        # Convert specs to actual Agent objects
        agents = swarm.create_agents_from_specs(agent_specs)

        print("✓ Created agents for research task:")
        for i, agent in enumerate(agents, 1):
            print(f"  Agent {i}:")
            print(f"    - Name: {agent.agent_name}")
            print(f"    - Description: {agent.agent_description}")
        print(f"✓ Created {len(agents)} agents successfully")
        return agents
    except Exception as e:
        print(f"✗ Agent creation test failed: {str(e)}")
        raise


def test_swarm_routing():
    """Test routing tasks through the swarm"""
    print_separator()
    print("Testing Swarm Routing")
    try:
        swarm = AutoSwarmBuilder(
            name="RouterTestSwarm",
            description="Testing routing capabilities",
        )
        agents = (
            test_agent_creation()
        )  # Get agents from previous test
        task = "Analyze the impact of AI on healthcare"

        print("Starting task routing...")
        result = swarm.initialize_swarm_router(agents, task)

        print("✓ Task routed successfully")
        print(
            f"  - Result length: {len(str(result)) if result else 0} characters"
        )
        print("✓ Swarm routing test passed")
        return result
    except Exception as e:
        print(f"✗ Swarm routing test failed: {str(e)}")
        raise


def test_full_swarm_execution():
    """Test complete swarm execution with a real task"""
    print_separator()
    print("Testing Full Swarm Execution")
    try:
        swarm = AutoSwarmBuilder(
            name="FullTestSwarm",
            description="Testing complete swarm functionality",
            max_loops=1,
        )
        task = (
            "Create a summary of recent advances in renewable energy"
        )

        print("Starting full swarm execution...")
        result = swarm.run(task)

        print("✓ Full swarm execution completed:")
        print(f"  - Output generated: {bool(result)}")
        print(
            f"  - Output length: {len(str(result)) if result else 0} characters"
        )
        print("✓ Full swarm execution test passed")
        return result
    except Exception as e:
        print(f"✗ Full swarm execution test failed: {str(e)}")
        raise


def test_error_handling():
    """Test error handling in swarm operations"""
    print_separator()
    print("Testing Error Handling")
    try:
        swarm = AutoSwarmBuilder()

        # Test with invalid agent configuration
        print("Testing invalid agent configuration...")
        try:
            swarm.create_agents_from_specs(
                {"agents": [{"agent_name": ""}]}
            )
            print(
                "✗ Should have raised an error for empty agent configuration"
            )
        except Exception as e:
            print(
                f"✓ Correctly handled invalid agent configuration: {type(e).__name__}"
            )

        # Test with None task
        print("\nTesting None task...")
        try:
            swarm.run(None)
            print("✗ Should have raised an error for None task")
        except Exception as e:
            print(
                f"✓ Correctly handled None task: {type(e).__name__}"
            )

        print("✓ Error handling test passed")
    except Exception as e:
        print(f"✗ Error handling test failed: {str(e)}")
        raise


def run_all_tests():
    """Run complete test suite"""
    print("\n=== Starting AutoSwarmBuilder Test Suite ===\n")

    try:
        # Run all tests in sequence
        test_initialization()
        test_agent_building()
        test_agent_creation()
        test_swarm_routing()
        test_full_swarm_execution()
        test_error_handling()

        print_separator()
        print("🎉 All tests completed successfully!")

    except Exception as e:
        print_separator()
        print(f"❌ Test suite failed: {str(e)}")
        raise


##############################################################################
# Bug Fix Tests (from test_auto_swarm_builder_fix.py) — bug #1115
##############################################################################


def test_create_agents_from_specs_with_dict():
    """Test that create_agents_from_specs handles dict input correctly."""
    builder = AutoSwarmBuilder()

    # Create specs as a dictionary
    specs = {
        "agents": [
            {
                "agent_name": "test_agent_1",
                "description": "Test agent 1 description",
                "system_prompt": "You are a helpful assistant",
                "model_name": "gpt-5.4",
                "max_loops": 1,
            }
        ]
    }

    agents = builder.create_agents_from_specs(specs)

    # Verify agents were created correctly
    assert len(agents) == 1
    assert isinstance(agents[0], Agent)
    assert agents[0].agent_name == "test_agent_1"

    # Verify description was mapped to agent_description
    assert hasattr(agents[0], "agent_description")
    assert agents[0].agent_description == "Test agent 1 description"


def test_create_agents_from_specs_with_pydantic():
    """Test that create_agents_from_specs handles Pydantic model input correctly.

    This is the main test for bug #1115 - it verifies that AgentSpec
    Pydantic models can be unpacked correctly.
    """
    builder = AutoSwarmBuilder()

    # Create specs as Pydantic AgentSpec objects
    agent_spec = AgentSpec(
        agent_name="test_agent_pydantic",
        description="Pydantic test agent",
        system_prompt="You are a helpful assistant",
        model_name="gpt-5.4",
        max_loops=1,
    )

    specs = {"agents": [agent_spec]}

    agents = builder.create_agents_from_specs(specs)

    # Verify agents were created correctly
    assert len(agents) == 1
    assert isinstance(agents[0], Agent)
    assert agents[0].agent_name == "test_agent_pydantic"

    # Verify description was mapped to agent_description
    assert hasattr(agents[0], "agent_description")
    assert agents[0].agent_description == "Pydantic test agent"


def test_parameter_name_mapping():
    """Test that 'description' field maps to 'agent_description' correctly."""
    builder = AutoSwarmBuilder()

    # Test with dict that has 'description'
    specs = {
        "agents": [
            {
                "agent_name": "mapping_test",
                "description": "This should map to agent_description",
                "system_prompt": "You are helpful",
            }
        ]
    }

    agents = builder.create_agents_from_specs(specs)

    assert len(agents) == 1
    agent = agents[0]

    # Verify description was mapped
    assert hasattr(agent, "agent_description")
    assert (
        agent.agent_description
        == "This should map to agent_description"
    )


def test_create_agents_from_specs_mixed_input():
    """Test that create_agents_from_specs handles mixed dict and Pydantic input."""
    builder = AutoSwarmBuilder()

    # Mix of dict and Pydantic objects
    dict_spec = {
        "agent_name": "dict_agent",
        "description": "Dict agent description",
        "system_prompt": "You are helpful",
    }

    pydantic_spec = AgentSpec(
        agent_name="pydantic_agent",
        description="Pydantic agent description",
        system_prompt="You are smart",
    )

    specs = {"agents": [dict_spec, pydantic_spec]}

    agents = builder.create_agents_from_specs(specs)

    # Verify both agents were created
    assert len(agents) == 2
    assert all(isinstance(agent, Agent) for agent in agents)

    # Verify both have correct descriptions
    dict_agent = next(
        a for a in agents if a.agent_name == "dict_agent"
    )
    pydantic_agent = next(
        a for a in agents if a.agent_name == "pydantic_agent"
    )

    assert dict_agent.agent_description == "Dict agent description"
    assert (
        pydantic_agent.agent_description
        == "Pydantic agent description"
    )


def test_set_random_models_for_agents_with_valid_agents():
    """Test set_random_models_for_agents with proper Agent objects."""
    # Create proper Agent objects
    agents = [
        Agent(
            agent_name="agent1",
            system_prompt="You are agent 1",
            max_loops=1,
        ),
        Agent(
            agent_name="agent2",
            system_prompt="You are agent 2",
            max_loops=1,
        ),
    ]

    # Set random models
    model_names = ["gpt-5.4", "gpt-4o", "claude-3-5-sonnet"]
    result = set_random_models_for_agents(
        agents=agents, model_names=model_names
    )

    # Verify results
    assert len(result) == 2
    assert all(isinstance(agent, Agent) for agent in result)
    assert all(hasattr(agent, "model_name") for agent in result)
    assert all(agent.model_name in model_names for agent in result)


def test_set_random_models_for_agents_with_single_agent():
    """Test set_random_models_for_agents with a single agent."""
    agent = Agent(
        agent_name="single_agent",
        system_prompt="You are helpful",
        max_loops=1,
    )

    model_names = ["gpt-5.4", "gpt-4o"]
    result = set_random_models_for_agents(
        agents=agent, model_names=model_names
    )

    assert isinstance(result, Agent)
    assert hasattr(result, "model_name")
    assert result.model_name in model_names


def test_set_random_models_for_agents_with_none():
    """Test set_random_models_for_agents with None returns random model name."""
    model_names = ["gpt-5.4", "gpt-4o", "claude-3-5-sonnet"]
    result = set_random_models_for_agents(
        agents=None, model_names=model_names
    )

    assert isinstance(result, str)
    assert result in model_names


@pytest.mark.skip(
    reason="This test requires API key and makes LLM calls"
)
def test_auto_swarm_builder_return_agents_objects_integration():
    """Integration test for AutoSwarmBuilder with swarm_type='return-agents-objects'.

    This test requires OPENAI_API_KEY and makes actual LLM calls.
    Run manually with: pytest -k test_auto_swarm_builder_return_agents_objects_integration -v
    """
    builder = AutoSwarmBuilder(
        swarm_type="return-agents-objects",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
    )

    agents = builder.run(
        "Create a team of 2 data analysis agents with specific roles"
    )

    # Verify agents were created
    assert isinstance(agents, list)
    assert len(agents) >= 1
    assert all(isinstance(agent, Agent) for agent in agents)
    assert all(hasattr(agent, "agent_name") for agent in agents)
    assert all(
        hasattr(agent, "agent_description") for agent in agents
    )


def test_agent_spec_to_agent_all_fields():
    """Test that all AgentSpec fields are properly passed to Agent."""
    builder = AutoSwarmBuilder()

    agent_spec = AgentSpec(
        agent_name="full_test_agent",
        description="Full test description",
        system_prompt="You are a comprehensive test agent",
        model_name="gpt-5.4",
        auto_generate_prompt=False,
        max_tokens=4096,
        temperature=0.7,
        role="worker",
        max_loops=3,
        goal="Test all parameters",
    )

    agents = builder.create_agents_from_specs(
        {"agents": [agent_spec]}
    )

    assert len(agents) == 1
    agent = agents[0]

    # Verify all fields were set
    assert agent.agent_name == "full_test_agent"
    assert agent.agent_description == "Full test description"
    # Agent may modify system_prompt by adding additional instructions
    assert "You are a comprehensive test agent" in agent.system_prompt
    assert agent.max_loops == 3
    assert agent.max_tokens == 4096
    assert agent.temperature == 0.7


def test_create_agents_from_specs_empty_list():
    """Test that create_agents_from_specs handles empty agent list."""
    builder = AutoSwarmBuilder()

    specs = {"agents": []}

    agents = builder.create_agents_from_specs(specs)

    assert isinstance(agents, list)
    assert len(agents) == 0


##############################################################################
# Scripted-LLM tests for the AutoSwarmBuilder redesign (no live LLM calls)
##############################################################################


class ScriptedLiteLLM:
    """Stand-in for LiteLLM: returns each entry in ``responses`` in order,
    one per ``.run()`` call, regardless of the prompt it's given."""

    def __init__(self, *args, responses=None, **kwargs):
        self._responses = list(responses or [])
        self.calls = 0

    def run(self, prompt):
        self.calls += 1
        return self._responses.pop(0)


def patch_scripted_llm(monkeypatch, responses):
    """Patch ``auto_swarm_builder.LiteLLM`` to return canned ``responses``,
    one per call, in order."""

    def factory(*args, **kwargs):
        return ScriptedLiteLLM(responses=responses)

    monkeypatch.setattr(
        "swarms.structs.auto_swarm_builder.LiteLLM", factory
    )


def minimal_router_config(**overrides):
    """A minimal valid SwarmRouterConfig payload, JSON-serializable."""
    payload = {
        "name": "Team",
        "description": "A team",
        "agents": [{"agent_name": "Solo"}],
        "swarm_type": "SequentialWorkflow",
        "rearrange_flow": None,
        "rules": None,
        "multi_agent_collab_prompt": None,
        "task": "placeholder",
    }
    payload.update(overrides)
    return payload


def test_build_llm_agent_uses_custom_system_prompt():
    """The constructor's system_prompt must actually reach the boss LLM."""
    builder = AutoSwarmBuilder(system_prompt="CUSTOM PROMPT")
    llm = builder.build_llm_agent(config=AgentSpec)
    assert llm.system_prompt == "CUSTOM PROMPT"


def test_markdown_fenced_json_is_parsed(monkeypatch):
    """LLM output wrapped in ```json fences should parse without a repair retry."""
    fenced = (
        "```json\n"
        '{"agents": [{"agent_name": "Researcher"}]}\n'
        "```"
    )
    patch_scripted_llm(monkeypatch, [fenced])

    builder = AutoSwarmBuilder(max_json_repair_attempts=0)
    result = builder.create_agents("some task")

    assert result["agents"][0]["agent_name"] == "Researcher"


def test_malformed_json_triggers_repair_retry(monkeypatch):
    """A malformed first response should be retried with a repair prompt."""
    patch_scripted_llm(
        monkeypatch,
        [
            "not json at all",
            json.dumps({"agents": [{"agent_name": "Fixed"}]}),
        ],
    )

    builder = AutoSwarmBuilder(max_json_repair_attempts=1)
    result = builder.create_agents("some task")

    assert result["agents"][0]["agent_name"] == "Fixed"


def test_exhausted_repair_attempts_raises(monkeypatch):
    """Repeated malformed JSON should raise once the retry budget is spent."""
    patch_scripted_llm(
        monkeypatch, ["still not json", "still not json"]
    )

    builder = AutoSwarmBuilder(max_json_repair_attempts=1)
    with pytest.raises(ValueError):
        builder.create_agents("some task")


def test_duplicate_agent_names_rejected_in_create_agents(
    monkeypatch,
):
    """The boss agent generating two agents with the same name must be rejected."""
    patch_scripted_llm(
        monkeypatch,
        [
            json.dumps(
                {
                    "agents": [
                        {"agent_name": "Dup"},
                        {"agent_name": "Dup"},
                    ]
                }
            )
        ],
    )

    builder = AutoSwarmBuilder(max_json_repair_attempts=0)
    with pytest.raises(ValueError, match="Duplicate agent_name"):
        builder.create_agents("some task")


def test_duplicate_agent_names_rejected_in_create_agents_from_specs():
    """create_agents_from_specs itself should reject duplicate names,
    independent of the LLM-generation path."""
    builder = AutoSwarmBuilder()
    specs = {
        "agents": [
            {"agent_name": "Same"},
            {"agent_name": "Same"},
        ]
    }
    with pytest.raises(ValueError, match="Duplicate agent_name"):
        builder.create_agents_from_specs(specs)


def test_max_agents_cap_enforced(monkeypatch):
    """Exceeding max_agents should raise instead of silently building
    an oversized team."""
    patch_scripted_llm(
        monkeypatch,
        [
            json.dumps(
                {
                    "agents": [
                        {"agent_name": "A"},
                        {"agent_name": "B"},
                        {"agent_name": "C"},
                    ]
                }
            )
        ],
    )

    builder = AutoSwarmBuilder(
        max_agents=2, max_json_repair_attempts=0
    )
    with pytest.raises(ValueError, match="max_agents"):
        builder.create_agents("some task")


def test_swarm_type_rejects_auto_swarm_builder_recursion(
    monkeypatch,
):
    """'AutoSwarmBuilder' would recurse into itself and must be rejected
    as a generated swarm_type."""
    patch_scripted_llm(
        monkeypatch,
        [
            json.dumps(
                minimal_router_config(swarm_type="AutoSwarmBuilder")
            )
        ],
    )

    builder = AutoSwarmBuilder(max_json_repair_attempts=0)
    with pytest.raises(ValueError, match="not usable here"):
        builder.create_router_config("some task")


def test_swarm_type_rejects_unrecognized_value(monkeypatch):
    """A hallucinated swarm_type outside the SwarmType literal must be
    rejected. Pydantic's Literal[SwarmType] field catches this during
    schema validation (inside _generate_structured_output) before
    _validate_swarm_type's policy check ever runs."""
    patch_scripted_llm(
        monkeypatch,
        [
            json.dumps(
                minimal_router_config(swarm_type="NotARealSwarmType")
            )
        ],
    )

    builder = AutoSwarmBuilder(max_json_repair_attempts=0)
    with pytest.raises(ValueError, match="swarm_type"):
        builder.create_router_config("some task")


def test_create_router_config_task_matches_caller(monkeypatch):
    """The returned 'task' must be the caller's exact task string, not
    whatever the LLM echoed back."""
    patch_scripted_llm(
        monkeypatch,
        [
            json.dumps(
                minimal_router_config(
                    task="something the LLM made up"
                )
            )
        ],
    )

    builder = AutoSwarmBuilder(max_json_repair_attempts=0)
    config = builder.create_router_config("the real task")

    assert config["task"] == "the real task"


def test_build_and_run_swarm_uses_single_consistent_spec(
    monkeypatch,
):
    """build_and_run_swarm must build agents from, and run the router
    with, the SAME spec — not a second independent LLM call."""
    patch_scripted_llm(
        monkeypatch,
        [
            json.dumps(
                minimal_router_config(
                    name="ResearchTeam",
                    agents=[
                        {"agent_name": "Researcher"},
                        {"agent_name": "Writer"},
                    ],
                    swarm_type="SequentialWorkflow",
                )
            )
        ],
    )

    captured = {}

    class FakeSwarmRouter:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, task):
            captured["run_task"] = task
            names = [a.agent_name for a in captured["agents"]]
            return f"ran with {names}"

    monkeypatch.setattr(
        "swarms.structs.auto_swarm_builder.SwarmRouter",
        FakeSwarmRouter,
    )

    builder = AutoSwarmBuilder(max_json_repair_attempts=0)
    result = builder.build_and_run_swarm("Research quantum computing")

    assert result["agents"] == ["Researcher", "Writer"]
    assert result["swarm_type"] == "SequentialWorkflow"
    assert result["task"] == "Research quantum computing"
    assert captured["run_task"] == "Research quantum computing"
    assert [a.agent_name for a in captured["agents"]] == [
        "Researcher",
        "Writer",
    ]
    # multi_agent_collab_prompt is str in SwarmRouterConfig but bool in
    # SwarmRouter — must not be forwarded as-is.
    assert "multi_agent_collab_prompt" not in captured


def test_run_execute_true_routes_to_build_and_run_swarm(
    monkeypatch,
):
    """run(task, execute=True) must take the build-and-run path even
    when swarm_type is left at its default."""
    patch_scripted_llm(
        monkeypatch, [json.dumps(minimal_router_config())]
    )
    monkeypatch.setattr(
        "swarms.structs.auto_swarm_builder.SwarmRouter",
        lambda **kwargs: type(
            "R", (), {"run": lambda self, task: "ok"}
        )(),
    )

    builder = AutoSwarmBuilder(max_json_repair_attempts=0)
    result = builder.run("some task", execute=True)

    assert isinstance(result, dict)
    assert result["output"] == "ok"


def test_auto_execute_default_used_when_execute_not_passed(
    monkeypatch,
):
    """When execute is omitted, run() should fall back to the
    auto_execute constructor default."""
    patch_scripted_llm(
        monkeypatch, [json.dumps(minimal_router_config())]
    )
    monkeypatch.setattr(
        "swarms.structs.auto_swarm_builder.SwarmRouter",
        lambda **kwargs: type(
            "R", (), {"run": lambda self, task: "ok"}
        )(),
    )

    builder = AutoSwarmBuilder(
        max_json_repair_attempts=0, auto_execute=True
    )
    result = builder.run("some task")

    assert isinstance(result, dict)
    assert result["output"] == "ok"


def test_dict_to_agent_delegates_to_create_agents_from_specs():
    """dict_to_agent should build real Agent objects via the same
    validated path as create_agents_from_specs."""
    builder = AutoSwarmBuilder()
    agents = builder.dict_to_agent(
        {"agents": [{"agent_name": "Delegate"}]}
    )
    assert len(agents) == 1
    assert agents[0].agent_name == "Delegate"


def test_dict_to_agent_handles_non_dict_input():
    """Non-dict input should return an empty list rather than raise."""
    builder = AutoSwarmBuilder()
    assert builder.dict_to_agent("not a dict") == []
    assert builder.dict_to_agent(None) == []


def test_reliability_check_rejects_non_positive_max_agents():
    with pytest.raises(ValueError, match="max_agents"):
        AutoSwarmBuilder(max_agents=0)


def test_reliability_check_rejects_negative_repair_attempts():
    with pytest.raises(ValueError, match="max_json_repair_attempts"):
        AutoSwarmBuilder(max_json_repair_attempts=-1)


def test_additional_llm_args_default_is_not_shared():
    """Guard against the mutable-default-argument bug: each instance's
    additional_llm_args must be an independent dict."""
    builder_a = AutoSwarmBuilder()
    builder_b = AutoSwarmBuilder()
    builder_a.additional_llm_args["foo"] = "bar"
    assert builder_b.additional_llm_args == {}


if __name__ == "__main__":
    run_all_tests()
