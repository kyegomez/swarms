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
    """Test building individual agents"""
    print_separator()
    print("Testing Agent Building")
    try:
        swarm = AutoSwarmBuilder()
        agent = swarm.build_agent(
            agent_name="TestAgent",
            agent_description="A test agent",
            agent_system_prompt="You are a test agent",
            max_loops=1,
        )

        print("✓ Built agent with configuration:")
        print(f"  - Name: {agent.agent_name}")
        print(f"  - Description: {agent.description}")
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
        agents = swarm._create_agents(task)

        print("✓ Created agents for research task:")
        for i, agent in enumerate(agents, 1):
            print(f"  Agent {i}:")
            print(f"    - Name: {agent.agent_name}")
            print(f"    - Description: {agent.description}")
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
        result = swarm.swarm_router(agents, task)

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
            swarm.build_agent("", "", "")
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


# Bug Fix Tests (from test_auto_swarm_builder_fix.py)
class TestAutoSwarmBuilderFix:
    """Tests for bug #1115 fix in AutoSwarmBuilder."""

    def test_create_agents_from_specs_with_dict(self):
        """Test that create_agents_from_specs handles dict input correctly."""
        builder = AutoSwarmBuilder()

        # Create specs as a dictionary
        specs = {
            "agents": [
                {
                    "agent_name": "test_agent_1",
                    "description": "Test agent 1 description",
                    "system_prompt": "You are a helpful assistant",
                    "model_name": "gpt-4o-mini",
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
        assert (
            agents[0].agent_description == "Test agent 1 description"
        )

    def test_create_agents_from_specs_with_pydantic(self):
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
            model_name="gpt-4o-mini",
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

    def test_parameter_name_mapping(self):
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

    def test_create_agents_from_specs_mixed_input(self):
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

        assert (
            dict_agent.agent_description == "Dict agent description"
        )
        assert (
            pydantic_agent.agent_description
            == "Pydantic agent description"
        )

    def test_set_random_models_for_agents_with_valid_agents(self):
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
        model_names = ["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet"]
        result = set_random_models_for_agents(
            agents=agents, model_names=model_names
        )

        # Verify results
        assert len(result) == 2
        assert all(isinstance(agent, Agent) for agent in result)
        assert all(hasattr(agent, "model_name") for agent in result)
        assert all(
            agent.model_name in model_names for agent in result
        )

    def test_set_random_models_for_agents_with_single_agent(self):
        """Test set_random_models_for_agents with a single agent."""
        agent = Agent(
            agent_name="single_agent",
            system_prompt="You are helpful",
            max_loops=1,
        )

        model_names = ["gpt-4o-mini", "gpt-4o"]
        result = set_random_models_for_agents(
            agents=agent, model_names=model_names
        )

        assert isinstance(result, Agent)
        assert hasattr(result, "model_name")
        assert result.model_name in model_names

    def test_set_random_models_for_agents_with_none(self):
        """Test set_random_models_for_agents with None returns random model name."""
        model_names = ["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet"]
        result = set_random_models_for_agents(
            agents=None, model_names=model_names
        )

        assert isinstance(result, str)
        assert result in model_names

    @pytest.mark.skip(
        reason="This test requires API key and makes LLM calls"
    )
    def test_auto_swarm_builder_return_agents_objects_integration(
        self,
    ):
        """Integration test for AutoSwarmBuilder with execution_type='return-agents-objects'.

        This test requires OPENAI_API_KEY and makes actual LLM calls.
        Run manually with: pytest -k test_auto_swarm_builder_return_agents_objects_integration -v
        """
        builder = AutoSwarmBuilder(
            execution_type="return-agents-objects",
            model_name="gpt-4o-mini",
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

    def test_agent_spec_to_agent_all_fields(self):
        """Test that all AgentSpec fields are properly passed to Agent."""
        builder = AutoSwarmBuilder()

        agent_spec = AgentSpec(
            agent_name="full_test_agent",
            description="Full test description",
            system_prompt="You are a comprehensive test agent",
            model_name="gpt-4o-mini",
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
        assert (
            "You are a comprehensive test agent"
            in agent.system_prompt
        )
        assert agent.max_loops == 3
        assert agent.max_tokens == 4096
        assert agent.temperature == 0.7

    def test_create_agents_from_specs_empty_list(self):
        """Test that create_agents_from_specs handles empty agent list."""
        builder = AutoSwarmBuilder()

        specs = {"agents": []}

        agents = builder.create_agents_from_specs(specs)

        assert isinstance(agents, list)
        assert len(agents) == 0


if __name__ == "__main__":
    run_all_tests()
