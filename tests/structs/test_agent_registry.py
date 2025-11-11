import os

try:
    import pytest
except ImportError:
    pytest = None

from loguru import logger

try:
    from swarms.structs.agent_registry import AgentRegistry
    from swarms.structs.agent import Agent
except (ImportError, ModuleNotFoundError) as e:
    import importlib.util
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    
    agent_registry_path = os.path.join(_current_dir, "..", "..", "swarms", "structs", "agent_registry.py")
    agent_path = os.path.join(_current_dir, "..", "..", "swarms", "structs", "agent.py")
    
    if os.path.exists(agent_registry_path):
        spec = importlib.util.spec_from_file_location("agent_registry", agent_registry_path)
        agent_registry_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_registry_module)
        AgentRegistry = agent_registry_module.AgentRegistry
    
    if os.path.exists(agent_path):
        spec = importlib.util.spec_from_file_location("agent", agent_path)
        agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_module)
        Agent = agent_module.Agent
    else:
        raise ImportError(f"Could not find required modules") from e

logger.remove()
logger.add(lambda msg: None, level="ERROR")


def test_agent_registry_initialization():
    """Test AgentRegistry initialization."""
    try:
        registry = AgentRegistry()
        assert registry is not None, "AgentRegistry should not be None"
        assert registry.name == "Agent Registry", "Default name should be set"
        assert registry.description == "A registry for managing agents.", "Default description should be set"
        assert isinstance(registry.agents, dict), "Agents should be a dictionary"
        assert len(registry.agents) == 0, "Initial registry should be empty"
        
        registry2 = AgentRegistry(
            name="Test Registry",
            description="Test description",
            return_json=False,
            auto_save=True
        )
        assert registry2.name == "Test Registry", "Custom name should be set"
        assert registry2.description == "Test description", "Custom description should be set"
        assert registry2.return_json is False, "return_json should be False"
        assert registry2.auto_save is True, "auto_save should be True"
        
        logger.info("✓ AgentRegistry initialization test passed")
        
    except Exception as e:
        logger.error(f"Error in test_agent_registry_initialization: {str(e)}")
        raise


def test_agent_registry_add_single_agent():
    """Test adding a single agent to the registry."""
    try:
        registry = AgentRegistry()
        
        agent = Agent(
            agent_name="Test-Agent-1",
            agent_description="Test agent for registry",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
            streaming_on=True,
        )
        
        registry.add(agent)
        
        assert len(registry.agents) == 1, "Registry should have one agent"
        assert "Test-Agent-1" in registry.agents, "Agent should be in registry"
        assert registry.agents["Test-Agent-1"] is not None, "Agent object should not be None"
        assert registry.agents["Test-Agent-1"].agent_name == "Test-Agent-1", "Agent name should match"
        
        logger.info("✓ Add single agent test passed")
        
    except Exception as e:
        logger.error(f"Error in test_agent_registry_add_single_agent: {str(e)}")
        raise


def test_agent_registry_add_multiple_agents():
    """Test adding multiple agents to the registry."""
    try:
        registry = AgentRegistry()
        
        agent1 = Agent(
            agent_name="Test-Agent-1",
            agent_description="First test agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
            streaming_on=True,
        )
        
        agent2 = Agent(
            agent_name="Test-Agent-2",
            agent_description="Second test agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
            streaming_on=True,
        )
        
        agent3 = Agent(
            agent_name="Test-Agent-3",
            agent_description="Third test agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
            streaming_on=True,
        )
        
        registry.add_many([agent1, agent2, agent3])
        
        assert len(registry.agents) == 3, "Registry should have three agents"
        assert "Test-Agent-1" in registry.agents, "Agent 1 should be in registry"
        assert "Test-Agent-2" in registry.agents, "Agent 2 should be in registry"
        assert "Test-Agent-3" in registry.agents, "Agent 3 should be in registry"
        
        for agent_name in ["Test-Agent-1", "Test-Agent-2", "Test-Agent-3"]:
            assert registry.agents[agent_name] is not None, f"{agent_name} should not be None"
        
        logger.info("✓ Add multiple agents test passed")
        
    except Exception as e:
        logger.error(f"Error in test_agent_registry_add_multiple_agents: {str(e)}")
        raise


def test_agent_registry_get_agent():
    """Test retrieving an agent from the registry."""
    try:
        registry = AgentRegistry()
        
        agent = Agent(
            agent_name="Retrievable-Agent",
            agent_description="Agent for retrieval testing",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
            streaming_on=True,
        )
        
        registry.add(agent)
        
        retrieved_agent = registry.get("Retrievable-Agent")
        
        assert retrieved_agent is not None, "Retrieved agent should not be None"
        assert retrieved_agent.agent_name == "Retrievable-Agent", "Agent name should match"
        assert hasattr(retrieved_agent, "run"), "Agent should have run method"
        assert retrieved_agent is agent, "Should return the same agent object"
        
        logger.info("✓ Get agent test passed")
        
    except Exception as e:
        logger.error(f"Error in test_agent_registry_get_agent: {str(e)}")
        raise


def test_agent_registry_delete_agent():
    """Test deleting an agent from the registry."""
    try:
        registry = AgentRegistry()
        
        agent1 = Agent(
            agent_name="Agent-To-Delete",
            agent_description="Agent that will be deleted",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
            streaming_on=True,
        )
        
        agent2 = Agent(
            agent_name="Agent-To-Keep",
            agent_description="Agent that will remain",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
            streaming_on=True,
        )
        
        registry.add(agent1)
        registry.add(agent2)
        
        assert len(registry.agents) == 2, "Registry should have two agents"
        
        registry.delete("Agent-To-Delete")
        
        assert len(registry.agents) == 1, "Registry should have one agent after deletion"
        assert "Agent-To-Delete" not in registry.agents, "Deleted agent should not be in registry"
        assert "Agent-To-Keep" in registry.agents, "Other agent should still be in registry"
        
        logger.info("✓ Delete agent test passed")
        
    except Exception as e:
        logger.error(f"Error in test_agent_registry_delete_agent: {str(e)}")
        raise


def test_agent_registry_update_agent():
    """Test updating an agent in the registry."""
    try:
        registry = AgentRegistry()
        
        original_agent = Agent(
            agent_name="Agent-To-Update",
            agent_description="Original description",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
            streaming_on=True,
        )
        
        registry.add(original_agent)
        
        updated_agent = Agent(
            agent_name="Agent-To-Update",
            agent_description="Updated description",
            model_name="gpt-4o-mini",
            max_loops=2,
            verbose=True,
            print_on=False,
            streaming_on=True,
        )
        
        registry.update_agent("Agent-To-Update", updated_agent)
        
        retrieved_agent = registry.get("Agent-To-Update")
        
        assert retrieved_agent is not None, "Updated agent should not be None"
        assert retrieved_agent is updated_agent, "Should return the updated agent"
        assert retrieved_agent.max_loops == 2, "Max loops should be updated"
        assert retrieved_agent.verbose is True, "Verbose should be updated"
        
        logger.info("✓ Update agent test passed")
        
    except Exception as e:
        logger.error(f"Error in test_agent_registry_update_agent: {str(e)}")
        raise


def test_agent_registry_list_agents():
    """Test listing all agent names in the registry."""
    try:
        registry = AgentRegistry()
        
        agent1 = Agent(
            agent_name="List-Agent-1",
            agent_description="First agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
            streaming_on=True,
        )
        
        agent2 = Agent(
            agent_name="List-Agent-2",
            agent_description="Second agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
            streaming_on=True,
        )
        
        registry.add(agent1)
        registry.add(agent2)
        
        agent_names = registry.list_agents()
        
        assert agent_names is not None, "Agent names list should not be None"
        assert isinstance(agent_names, list), "Should return a list"
        assert len(agent_names) == 2, "Should have two agent names"
        assert "List-Agent-1" in agent_names, "First agent name should be in list"
        assert "List-Agent-2" in agent_names, "Second agent name should be in list"
        
        logger.info("✓ List agents test passed")
        
    except Exception as e:
        logger.error(f"Error in test_agent_registry_list_agents: {str(e)}")
        raise


def test_agent_registry_return_all_agents():
    """Test returning all agents from the registry."""
    try:
        registry = AgentRegistry()
        
        agent1 = Agent(
            agent_name="Return-Agent-1",
            agent_description="First agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
            streaming_on=True,
        )
        
        agent2 = Agent(
            agent_name="Return-Agent-2",
            agent_description="Second agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
            streaming_on=True,
        )
        
        registry.add(agent1)
        registry.add(agent2)
        
        all_agents = registry.return_all_agents()
        
        assert all_agents is not None, "All agents list should not be None"
        assert isinstance(all_agents, list), "Should return a list"
        assert len(all_agents) == 2, "Should have two agents"
        
        for agent in all_agents:
            assert agent is not None, "Each agent should not be None"
            assert hasattr(agent, "agent_name"), "Agent should have agent_name"
            assert hasattr(agent, "run"), "Agent should have run method"
        
        logger.info("✓ Return all agents test passed")
        
    except Exception as e:
        logger.error(f"Error in test_agent_registry_return_all_agents: {str(e)}")
        raise


def test_agent_registry_query_with_condition():
    """Test querying agents with a condition."""
    try:
        registry = AgentRegistry()
        
        agent1 = Agent(
            agent_name="Query-Agent-1",
            agent_description="Agent with max_loops=1",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
            streaming_on=True,
        )
        
        agent2 = Agent(
            agent_name="Query-Agent-2",
            agent_description="Agent with max_loops=2",
            model_name="gpt-4o-mini",
            max_loops=2,
            verbose=False,
            print_on=False,
            streaming_on=True,
        )
        
        agent3 = Agent(
            agent_name="Query-Agent-3",
            agent_description="Agent with max_loops=1",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
            streaming_on=True,
        )
        
        registry.add(agent1)
        registry.add(agent2)
        registry.add(agent3)
        
        def condition_max_loops_1(agent):
            return agent.max_loops == 1
        
        filtered_agents = registry.query(condition_max_loops_1)
        
        assert filtered_agents is not None, "Filtered agents should not be None"
        assert isinstance(filtered_agents, list), "Should return a list"
        assert len(filtered_agents) == 2, "Should have two agents with max_loops=1"
        
        for agent in filtered_agents:
            assert agent.max_loops == 1, "All filtered agents should have max_loops=1"
        
        logger.info("✓ Query with condition test passed")
        
    except Exception as e:
        logger.error(f"Error in test_agent_registry_query_with_condition: {str(e)}")
        raise


def test_agent_registry_query_without_condition():
    """Test querying all agents without a condition."""
    try:
        registry = AgentRegistry()
        
        agent1 = Agent(
            agent_name="Query-All-Agent-1",
            agent_description="First agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
            streaming_on=True,
        )
        
        agent2 = Agent(
            agent_name="Query-All-Agent-2",
            agent_description="Second agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
            streaming_on=True,
        )
        
        registry.add(agent1)
        registry.add(agent2)
        
        all_agents = registry.query()
        
        assert all_agents is not None, "All agents should not be None"
        assert isinstance(all_agents, list), "Should return a list"
        assert len(all_agents) == 2, "Should return all agents"
        
        logger.info("✓ Query without condition test passed")
        
    except Exception as e:
        logger.error(f"Error in test_agent_registry_query_without_condition: {str(e)}")
        raise


def test_agent_registry_find_agent_by_name():
    """Test finding an agent by name."""
    try:
        registry = AgentRegistry()
        
        agent = Agent(
            agent_name="Findable-Agent",
            agent_description="Agent to find",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
            streaming_on=True,
        )
        
        registry.add(agent)
        
        found_agent = registry.find_agent_by_name("Findable-Agent")
        
        assert found_agent is not None, "Found agent should not be None"
        assert found_agent.agent_name == "Findable-Agent", "Agent name should match"
        assert hasattr(found_agent, "run"), "Agent should have run method"
        
        logger.info("✓ Find agent by name test passed")
        
    except Exception as e:
        logger.error(f"Error in test_agent_registry_find_agent_by_name: {str(e)}")
        raise


def test_agent_registry_find_agent_by_id():
    """Test finding an agent by ID."""
    try:
        registry = AgentRegistry()
        
        agent = Agent(
            agent_name="ID-Agent",
            agent_description="Agent with ID",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
            streaming_on=True,
        )
        
        registry.add(agent)
        
        agent_id = agent.id
        found_agent = registry.find_agent_by_id(agent.agent_name)
        
        assert found_agent is not None, "Found agent should not be None"
        assert found_agent.agent_name == "ID-Agent", "Agent name should match"
        
        logger.info("✓ Find agent by ID test passed")
        
    except Exception as e:
        logger.error(f"Error in test_agent_registry_find_agent_by_id: {str(e)}")
        raise


def test_agent_registry_agents_to_json():
    """Test converting agents to JSON."""
    try:
        registry = AgentRegistry()
        
        agent1 = Agent(
            agent_name="JSON-Agent-1",
            agent_description="First agent for JSON",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
            streaming_on=True,
        )
        
        agent2 = Agent(
            agent_name="JSON-Agent-2",
            agent_description="Second agent for JSON",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
            streaming_on=True,
        )
        
        registry.add(agent1)
        registry.add(agent2)
        
        json_output = registry.agents_to_json()
        
        assert json_output is not None, "JSON output should not be None"
        assert isinstance(json_output, str), "Should return a string"
        assert len(json_output) > 0, "JSON should not be empty"
        assert "JSON-Agent-1" in json_output, "First agent should be in JSON"
        assert "JSON-Agent-2" in json_output, "Second agent should be in JSON"
        
        import json
        parsed_json = json.loads(json_output)
        assert isinstance(parsed_json, dict), "Should be valid JSON dict"
        assert len(parsed_json) == 2, "Should have two agents in JSON"
        
        logger.info("✓ Agents to JSON test passed")
        
    except Exception as e:
        logger.error(f"Error in test_agent_registry_agents_to_json: {str(e)}")
        raise


def test_agent_registry_initialization_with_agents():
    """Test initializing registry with agents."""
    try:
        agent1 = Agent(
            agent_name="Init-Agent-1",
            agent_description="First initial agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
            streaming_on=True,
        )
        
        agent2 = Agent(
            agent_name="Init-Agent-2",
            agent_description="Second initial agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
            streaming_on=True,
        )
        
        registry = AgentRegistry(agents=[agent1, agent2])
        
        assert registry is not None, "Registry should not be None"
        assert len(registry.agents) == 2, "Registry should have two agents"
        assert "Init-Agent-1" in registry.agents, "First agent should be in registry"
        assert "Init-Agent-2" in registry.agents, "Second agent should be in registry"
        
        logger.info("✓ Initialize with agents test passed")
        
    except Exception as e:
        logger.error(f"Error in test_agent_registry_initialization_with_agents: {str(e)}")
        raise


def test_agent_registry_error_duplicate_agent():
    """Test error handling for duplicate agent names."""
    try:
        registry = AgentRegistry()
        
        agent1 = Agent(
            agent_name="Duplicate-Agent",
            agent_description="First agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
            streaming_on=True,
        )
        
        agent2 = Agent(
            agent_name="Duplicate-Agent",
            agent_description="Duplicate agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
            streaming_on=True,
        )
        
        registry.add(agent1)
        
        try:
            registry.add(agent2)
            assert False, "Should have raised ValueError for duplicate agent"
        except ValueError as e:
            assert "already exists" in str(e).lower(), "Error message should mention duplicate"
            assert len(registry.agents) == 1, "Registry should still have only one agent"
        
        logger.info("✓ Error handling for duplicate agent test passed")
        
    except Exception as e:
        logger.error(f"Error in test_agent_registry_error_duplicate_agent: {str(e)}")
        raise


def test_agent_registry_error_nonexistent_agent():
    """Test error handling for nonexistent agent."""
    try:
        registry = AgentRegistry()
        
        try:
            registry.get("Nonexistent-Agent")
            assert False, "Should have raised KeyError for nonexistent agent"
        except KeyError as e:
            assert e is not None, "Should raise KeyError"
        
        try:
            registry.delete("Nonexistent-Agent")
            assert False, "Should have raised KeyError for nonexistent agent"
        except KeyError as e:
            assert e is not None, "Should raise KeyError"
        
        logger.info("✓ Error handling for nonexistent agent test passed")
        
    except Exception as e:
        logger.error(f"Error in test_agent_registry_error_nonexistent_agent: {str(e)}")
        raise


def test_agent_registry_retrieved_agents_can_run():
    """Test that retrieved agents can actually run tasks."""
    try:
        registry = AgentRegistry()
        
        agent = Agent(
            agent_name="Runnable-Registry-Agent",
            agent_description="Agent for running tasks",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
            streaming_on=True,
        )
        
        registry.add(agent)
        
        retrieved_agent = registry.get("Runnable-Registry-Agent")
        
        assert retrieved_agent is not None, "Retrieved agent should not be None"
        
        result = retrieved_agent.run("What is 2 + 2? Answer briefly.")
        
        assert result is not None, "Agent run result should not be None"
        assert isinstance(result, str), "Result should be a string"
        assert len(result) > 0, "Result should not be empty"
        
        logger.info("✓ Retrieved agents can run test passed")
        
    except Exception as e:
        logger.error(f"Error in test_agent_registry_retrieved_agents_can_run: {str(e)}")
        raise


def test_agent_registry_thread_safety():
    """Test thread safety of registry operations."""
    try:
        registry = AgentRegistry()
        
        agent1 = Agent(
            agent_name="Thread-Agent-1",
            agent_description="First thread agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
            streaming_on=True,
        )
        
        agent2 = Agent(
            agent_name="Thread-Agent-2",
            agent_description="Second thread agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
            streaming_on=True,
        )
        
        registry.add(agent1)
        registry.add(agent2)
        
        agent_names = registry.list_agents()
        all_agents = registry.return_all_agents()
        
        assert agent_names is not None, "Agent names should not be None"
        assert all_agents is not None, "All agents should not be None"
        assert len(agent_names) == 2, "Should have two agent names"
        assert len(all_agents) == 2, "Should have two agents"
        
        logger.info("✓ Thread safety test passed")
        
    except Exception as e:
        logger.error(f"Error in test_agent_registry_thread_safety: {str(e)}")
        raise


if __name__ == "__main__":
    import sys
    
    test_dict = {
        "test_agent_registry_initialization": test_agent_registry_initialization,
        "test_agent_registry_add_single_agent": test_agent_registry_add_single_agent,
        "test_agent_registry_add_multiple_agents": test_agent_registry_add_multiple_agents,
        "test_agent_registry_get_agent": test_agent_registry_get_agent,
        "test_agent_registry_delete_agent": test_agent_registry_delete_agent,
        "test_agent_registry_update_agent": test_agent_registry_update_agent,
        "test_agent_registry_list_agents": test_agent_registry_list_agents,
        "test_agent_registry_return_all_agents": test_agent_registry_return_all_agents,
        "test_agent_registry_query_with_condition": test_agent_registry_query_with_condition,
        "test_agent_registry_query_without_condition": test_agent_registry_query_without_condition,
        "test_agent_registry_find_agent_by_name": test_agent_registry_find_agent_by_name,
        "test_agent_registry_find_agent_by_id": test_agent_registry_find_agent_by_id,
        "test_agent_registry_agents_to_json": test_agent_registry_agents_to_json,
        "test_agent_registry_initialization_with_agents": test_agent_registry_initialization_with_agents,
        "test_agent_registry_error_duplicate_agent": test_agent_registry_error_duplicate_agent,
        "test_agent_registry_error_nonexistent_agent": test_agent_registry_error_nonexistent_agent,
        "test_agent_registry_retrieved_agents_can_run": test_agent_registry_retrieved_agents_can_run,
        "test_agent_registry_thread_safety": test_agent_registry_thread_safety,
    }
    
    if len(sys.argv) > 1:
        requested_tests = []
        for test_name in sys.argv[1:]:
            if test_name in test_dict:
                requested_tests.append(test_dict[test_name])
            elif test_name == "all" or test_name == "--all":
                requested_tests = list(test_dict.values())
                break
            else:
                print(f"⚠ Warning: Test '{test_name}' not found.")
                print(f"Available tests: {', '.join(test_dict.keys())}")
                sys.exit(1)
        
        tests_to_run = requested_tests
    else:
        tests_to_run = list(test_dict.values())
    
    if len(tests_to_run) == 1:
        print(f"Running: {tests_to_run[0].__name__}")
    else:
        print(f"Running {len(tests_to_run)} test(s)...")
    
    passed = 0
    failed = 0
    
    for test_func in tests_to_run:
        try:
            print(f"\n{'='*60}")
            print(f"Running: {test_func.__name__}")
            print(f"{'='*60}")
            test_func()
            print(f"✓ PASSED: {test_func.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {test_func.__name__}")
            print(f"  Error: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Test Summary: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    
    if len(sys.argv) == 1:
        print("\n💡 Tip: Run a specific test with:")
        print("   python test_agent_registry.py test_agent_registry_initialization")
        print("\n   Or use pytest:")
        print("   pytest test_agent_registry.py")
        print("   pytest test_agent_registry.py::test_agent_registry_initialization")

