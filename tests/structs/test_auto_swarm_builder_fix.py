"""
Tests for bug #1115 fix in AutoSwarmBuilder.

This test module verifies the fix for AttributeError when creating agents
from AgentSpec Pydantic models in AutoSwarmBuilder.

Bug: https://github.com/kyegomez/swarms/issues/1115
"""

import pytest
from pydantic import BaseModel

from swarms.structs.agent import Agent
from swarms.structs.auto_swarm_builder import (
    AgentSpec,
    AutoSwarmBuilder,
)
from swarms.structs.ma_utils import set_random_models_for_agents


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

    def test_set_random_models_for_agents_with_valid_agents(
        self,
    ):
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

    def test_set_random_models_for_agents_with_single_agent(
        self,
    ):
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
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
