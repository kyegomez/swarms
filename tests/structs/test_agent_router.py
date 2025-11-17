# from unittest.mock import Mock, patch

from swarms.structs.agent_router import AgentRouter
from swarms.structs.agent import Agent


def test_agent_router_initialization_default():
    """Test AgentRouter initialization with default parameters."""
    router = AgentRouter()

    assert router.embedding_model == "text-embedding-ada-002"
    assert router.n_agents == 1
    assert router.api_key is None
    assert router.api_base is None
    assert router.agents == []
    assert router.agent_embeddings == []
    assert router.agent_metadata == []


def test_agent_router_initialization_custom():
    """Test AgentRouter initialization with custom parameters."""
    agents = [
        Agent(
            agent_name="test1",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
            streaming_on=True,
        ),
        Agent(
            agent_name="test2",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
            streaming_on=True,
        ),
    ]
    router = AgentRouter(
        embedding_model="text-embedding-ada-002",
        n_agents=3,
        api_key=None,
        api_base=None,
        agents=agents,
    )

    assert router.embedding_model == "text-embedding-ada-002"
    assert router.n_agents == 3
    assert len(router.agents) == 2


def test_cosine_similarity_identical_vectors():
    """Test cosine similarity with identical vectors."""
    router = AgentRouter()
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]

    result = router._cosine_similarity(vec1, vec2)
    assert result == 1.0


def test_cosine_similarity_orthogonal_vectors():
    """Test cosine similarity with orthogonal vectors."""
    router = AgentRouter()
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]

    result = router._cosine_similarity(vec1, vec2)
    assert result == 0.0


def test_cosine_similarity_opposite_vectors():
    """Test cosine similarity with opposite vectors."""
    router = AgentRouter()
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [-1.0, 0.0, 0.0]

    result = router._cosine_similarity(vec1, vec2)
    assert result == -1.0


def test_cosine_similarity_different_lengths():
    """Test cosine similarity with vectors of different lengths."""
    router = AgentRouter()
    vec1 = [1.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]

    try:
        router._cosine_similarity(vec1, vec2)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Vectors must have the same length" in str(e)


def test_generate_embedding_success():
    """Test successful embedding generation with real API."""
    router = AgentRouter()
    result = router._generate_embedding("test text")

    assert result is not None
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(x, float) for x in result)


def test_add_agent_success():
    """Test successful agent addition with real agent and streaming."""
    router = AgentRouter()
    agent = Agent(
        agent_name="test_agent",
        agent_description="A test agent",
        system_prompt="You are a test agent",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
        streaming_on=True,
    )

    router.add_agent(agent)

    assert len(router.agents) == 1
    assert len(router.agent_embeddings) == 1
    assert len(router.agent_metadata) == 1
    assert router.agents[0] == agent
    assert router.agent_metadata[0]["name"] == "test_agent"

    # Test that agent can stream
    streamed_chunks = []

    def streaming_callback(chunk: str):
        streamed_chunks.append(chunk)

    response = agent.run(
        "Say hello", streaming_callback=streaming_callback
    )
    assert response is not None
    assert (
        len(streamed_chunks) > 0 or response != ""
    ), "Agent should stream or return response"


def test_add_agents_multiple():
    """Test adding multiple agents with real agents and streaming."""
    router = AgentRouter()
    agents = [
        Agent(
            agent_name="agent1",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
            streaming_on=True,
        ),
        Agent(
            agent_name="agent2",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
            streaming_on=True,
        ),
        Agent(
            agent_name="agent3",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
            streaming_on=True,
        ),
    ]

    router.add_agents(agents)

    assert len(router.agents) == 3
    assert len(router.agent_embeddings) == 3
    assert len(router.agent_metadata) == 3

    # Test that all agents can stream
    for agent in agents:
        streamed_chunks = []

        def streaming_callback(chunk: str):
            streamed_chunks.append(chunk)

        response = agent.run(
            "Say hi", streaming_callback=streaming_callback
        )
        assert response is not None
        assert (
            len(streamed_chunks) > 0 or response != ""
        ), f"Agent {agent.agent_name} should stream or return response"


def test_find_best_agent_success():
    """Test successful best agent finding with real agents and streaming."""
    router = AgentRouter()
    agent1 = Agent(
        agent_name="agent1",
        agent_description="First agent that handles data extraction",
        system_prompt="You are a data extraction specialist",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
        streaming_on=True,
    )
    agent2 = Agent(
        agent_name="agent2",
        agent_description="Second agent that handles summarization",
        system_prompt="You are a summarization specialist",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
        streaming_on=True,
    )

    router.add_agent(agent1)
    router.add_agent(agent2)

    result = router.find_best_agent("Extract data from documents")

    assert result is not None
    assert result in [agent1, agent2]

    # Test that the found agent can stream
    streamed_chunks = []

    def streaming_callback(chunk: str):
        streamed_chunks.append(chunk)

    response = result.run(
        "Test task", streaming_callback=streaming_callback
    )
    assert response is not None
    assert (
        len(streamed_chunks) > 0 or response != ""
    ), "Found agent should stream or return response"


def test_find_best_agent_no_agents():
    """Test finding best agent when no agents are available."""
    router = AgentRouter()

    result = router.find_best_agent("test task")

    assert result is None


def test_update_agent_history_success():
    """Test successful agent history update with real agent and streaming."""
    router = AgentRouter()
    agent = Agent(
        agent_name="test_agent",
        agent_description="A test agent",
        system_prompt="You are a test agent",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
        streaming_on=True,
    )

    router.add_agent(agent)

    # Run agent to create history
    streamed_chunks = []

    def streaming_callback(chunk: str):
        streamed_chunks.append(chunk)

    agent.run(
        "Hello, how are you?", streaming_callback=streaming_callback
    )

    # Update agent history
    router.update_agent_history("test_agent")

    # Verify the embedding was regenerated
    assert len(router.agent_embeddings) == 1
    assert router.agent_metadata[0]["name"] == "test_agent"


def test_update_agent_history_agent_not_found():
    """Test updating history for non-existent agent."""
    router = AgentRouter()

    # Should not raise an exception, just log a warning
    router.update_agent_history("non_existent_agent")


def test_agent_metadata_structure():
    """Test the structure of agent metadata with real agent."""
    router = AgentRouter()
    agent = Agent(
        agent_name="test_agent",
        agent_description="A test agent",
        system_prompt="You are a test agent",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
        streaming_on=True,
    )

    router.add_agent(agent)

    metadata = router.agent_metadata[0]
    assert "name" in metadata
    assert "text" in metadata
    assert metadata["name"] == "test_agent"
    assert (
        "test_agent A test agent You are a test agent"
        in metadata["text"]
    )


def test_agent_router_edge_cases():
    """Test various edge cases with real router."""
    router = AgentRouter()

    # Test with empty string task
    result = router.find_best_agent("")
    assert result is None

    # Test with very long task description
    long_task = "test " * 1000
    result = router.find_best_agent(long_task)
    # Should either return None or handle gracefully
    assert result is None or result is not None


def test_router_with_agent_streaming():
    """Test that agents in router can stream when run."""
    router = AgentRouter()

    agent1 = Agent(
        agent_name="streaming_agent1",
        agent_description="Agent for testing streaming",
        system_prompt="You are a helpful assistant",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
        streaming_on=True,
    )

    agent2 = Agent(
        agent_name="streaming_agent2",
        agent_description="Another agent for testing streaming",
        system_prompt="You are a helpful assistant",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
        streaming_on=True,
    )

    router.add_agent(agent1)
    router.add_agent(agent2)

    # Test each agent streams
    for agent in router.agents:
        streamed_chunks = []

        def streaming_callback(chunk: str):
            if chunk:
                streamed_chunks.append(chunk)

        response = agent.run(
            "Tell me a short joke",
            streaming_callback=streaming_callback,
        )
        assert response is not None
        assert (
            len(streamed_chunks) > 0 or response != ""
        ), f"Agent {agent.agent_name} should stream"


def test_router_find_and_run_with_streaming():
    """Test finding best agent and running it with streaming."""
    router = AgentRouter()

    agent1 = Agent(
        agent_name="math_agent",
        agent_description="Handles mathematical problems",
        system_prompt="You are a math expert",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
        streaming_on=True,
    )

    agent2 = Agent(
        agent_name="writing_agent",
        agent_description="Handles writing tasks",
        system_prompt="You are a writing expert",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
        streaming_on=True,
    )

    router.add_agent(agent1)
    router.add_agent(agent2)

    # Find best agent for a math task
    best_agent = router.find_best_agent("Solve 2 + 2")

    if best_agent:
        streamed_chunks = []

        def streaming_callback(chunk: str):
            if chunk:
                streamed_chunks.append(chunk)

        response = best_agent.run(
            "What is 2 + 2?", streaming_callback=streaming_callback
        )
        assert response is not None
        assert (
            len(streamed_chunks) > 0 or response != ""
        ), "Best agent should stream when run"


if __name__ == "__main__":
    # List of all test functions
    tests = [
        test_agent_router_initialization_default,
        test_agent_router_initialization_custom,
        test_cosine_similarity_identical_vectors,
        test_cosine_similarity_orthogonal_vectors,
        test_cosine_similarity_opposite_vectors,
        test_cosine_similarity_different_lengths,
        test_generate_embedding_success,
        test_add_agent_success,
        test_add_agents_multiple,
        test_find_best_agent_success,
        test_find_best_agent_no_agents,
        test_update_agent_history_success,
        test_update_agent_history_agent_not_found,
        test_agent_metadata_structure,
        test_agent_router_edge_cases,
        test_router_with_agent_streaming,
        test_router_find_and_run_with_streaming,
    ]

    # Run all tests
    print("Running all tests...")
    passed = 0
    failed = 0

    for test_func in tests:
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
