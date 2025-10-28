import pytest
from unittest.mock import Mock, patch

from swarms.structs.agent_router import AgentRouter
from swarms.structs.agent import Agent


@pytest.fixture
def test_agent():
    """Create a real agent for testing."""
    with patch("swarms.structs.agent.LiteLLM") as mock_llm:
        mock_llm.return_value.run.return_value = "Test response"
        return Agent(
            agent_name="test_agent",
            agent_description="A test agent",
            system_prompt="You are a test agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
        )


def test_agent_router_initialization_default():
    """Test AgentRouter initialization with default parameters."""
    with patch("swarms.structs.agent_router.embedding"):
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
    with patch("swarms.structs.agent_router.embedding"), patch(
        "swarms.structs.agent.LiteLLM"
    ) as mock_llm:
        mock_llm.return_value.run.return_value = "Test response"
        agents = [
            Agent(
                agent_name="test1",
                model_name="gpt-4o-mini",
                max_loops=1,
                verbose=False,
                print_on=False,
            ),
            Agent(
                agent_name="test2",
                model_name="gpt-4o-mini",
                max_loops=1,
                verbose=False,
                print_on=False,
            ),
        ]
        router = AgentRouter(
            embedding_model="custom-model",
            n_agents=3,
            api_key="custom_key",
            api_base="custom_base",
            agents=agents,
        )

        assert router.embedding_model == "custom-model"
        assert router.n_agents == 3
        assert router.api_key == "custom_key"
        assert router.api_base == "custom_base"
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

    with pytest.raises(
        ValueError, match="Vectors must have the same length"
    ):
        router._cosine_similarity(vec1, vec2)


@patch("swarms.structs.agent_router.embedding")
def test_generate_embedding_success(mock_embedding):
    """Test successful embedding generation."""
    mock_embedding.return_value.data = [
        Mock(embedding=[0.1, 0.2, 0.3, 0.4])
    ]

    router = AgentRouter()
    result = router._generate_embedding("test text")

    assert result == [0.1, 0.2, 0.3, 0.4]
    mock_embedding.assert_called_once()


@patch("swarms.structs.agent_router.embedding")
def test_generate_embedding_error(mock_embedding):
    """Test embedding generation error handling."""
    mock_embedding.side_effect = Exception("API Error")

    router = AgentRouter()

    with pytest.raises(Exception, match="API Error"):
        router._generate_embedding("test text")


@patch("swarms.structs.agent_router.embedding")
def test_add_agent_success(mock_embedding, test_agent):
    """Test successful agent addition."""
    mock_embedding.return_value.data = [
        Mock(embedding=[0.1, 0.2, 0.3])
    ]

    router = AgentRouter()
    router.add_agent(test_agent)

    assert len(router.agents) == 1
    assert len(router.agent_embeddings) == 1
    assert len(router.agent_metadata) == 1
    assert router.agents[0] == test_agent
    assert router.agent_embeddings[0] == [0.1, 0.2, 0.3]
    assert router.agent_metadata[0]["name"] == "test_agent"


@patch("swarms.structs.agent_router.embedding")
def test_add_agent_retry_error(mock_embedding, test_agent):
    """Test agent addition with retry mechanism failure."""
    mock_embedding.side_effect = Exception("Embedding error")

    router = AgentRouter()

    # Should raise RetryError after retries are exhausted
    with pytest.raises(Exception) as exc_info:
        router.add_agent(test_agent)

    # Check that it's a retry error or contains the original error
    assert "Embedding error" in str(
        exc_info.value
    ) or "RetryError" in str(exc_info.value)


@patch("swarms.structs.agent_router.embedding")
def test_add_agents_multiple(mock_embedding):
    """Test adding multiple agents."""
    mock_embedding.return_value.data = [
        Mock(embedding=[0.1, 0.2, 0.3])
    ]

    with patch("swarms.structs.agent.LiteLLM") as mock_llm:
        mock_llm.return_value.run.return_value = "Test response"
        router = AgentRouter()
        agents = [
            Agent(
                agent_name="agent1",
                model_name="gpt-4o-mini",
                max_loops=1,
                verbose=False,
                print_on=False,
            ),
            Agent(
                agent_name="agent2",
                model_name="gpt-4o-mini",
                max_loops=1,
                verbose=False,
                print_on=False,
            ),
            Agent(
                agent_name="agent3",
                model_name="gpt-4o-mini",
                max_loops=1,
                verbose=False,
                print_on=False,
            ),
        ]

        router.add_agents(agents)

        assert len(router.agents) == 3
        assert len(router.agent_embeddings) == 3
        assert len(router.agent_metadata) == 3


@patch("swarms.structs.agent_router.embedding")
def test_find_best_agent_success(mock_embedding):
    """Test successful best agent finding."""
    # Mock embeddings for agents and task
    mock_embedding.side_effect = [
        Mock(data=[Mock(embedding=[0.1, 0.2, 0.3])]),  # agent1
        Mock(data=[Mock(embedding=[0.4, 0.5, 0.6])]),  # agent2
        Mock(data=[Mock(embedding=[0.7, 0.8, 0.9])]),  # task
    ]

    with patch("swarms.structs.agent.LiteLLM") as mock_llm:
        mock_llm.return_value.run.return_value = "Test response"
        router = AgentRouter()
        agent1 = Agent(
            agent_name="agent1",
            agent_description="First agent",
            system_prompt="Prompt 1",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
        )
        agent2 = Agent(
            agent_name="agent2",
            agent_description="Second agent",
            system_prompt="Prompt 2",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
        )

        router.add_agent(agent1)
        router.add_agent(agent2)

        # Mock the similarity calculation to return predictable results
        with patch.object(
            router, "_cosine_similarity"
        ) as mock_similarity:
            mock_similarity.side_effect = [
                0.8,
                0.6,
            ]  # agent1 more similar

            result = router.find_best_agent("test task")

            assert result == agent1


def test_find_best_agent_no_agents():
    """Test finding best agent when no agents are available."""
    with patch("swarms.structs.agent_router.embedding"):
        router = AgentRouter()

        result = router.find_best_agent("test task")

        assert result is None


@patch("swarms.structs.agent_router.embedding")
def test_find_best_agent_retry_error(mock_embedding):
    """Test error handling in find_best_agent with retry mechanism."""
    mock_embedding.side_effect = Exception("API Error")

    with patch("swarms.structs.agent.LiteLLM") as mock_llm:
        mock_llm.return_value.run.return_value = "Test response"
        router = AgentRouter()
        router.agents = [
            Agent(
                agent_name="agent1",
                model_name="gpt-4o-mini",
                max_loops=1,
                verbose=False,
                print_on=False,
            )
        ]
        router.agent_embeddings = [[0.1, 0.2, 0.3]]

        # Should raise RetryError after retries are exhausted
        with pytest.raises(Exception) as exc_info:
            router.find_best_agent("test task")

        # Check that it's a retry error or contains the original error
        assert "API Error" in str(
            exc_info.value
        ) or "RetryError" in str(exc_info.value)


@patch("swarms.structs.agent_router.embedding")
def test_update_agent_history_success(mock_embedding, test_agent):
    """Test successful agent history update."""
    mock_embedding.return_value.data = [
        Mock(embedding=[0.1, 0.2, 0.3])
    ]

    router = AgentRouter()
    router.add_agent(test_agent)

    # Update agent history
    router.update_agent_history("test_agent")

    # Verify the embedding was regenerated
    assert (
        mock_embedding.call_count == 2
    )  # Once for add, once for update


def test_update_agent_history_agent_not_found():
    """Test updating history for non-existent agent."""
    with patch(
        "swarms.structs.agent_router.embedding"
    ) as mock_embedding:
        mock_embedding.return_value.data = [
            Mock(embedding=[0.1, 0.2, 0.3])
        ]
        router = AgentRouter()

        # Should not raise an exception, just log a warning
        router.update_agent_history("non_existent_agent")


@patch("swarms.structs.agent_router.embedding")
def test_agent_metadata_structure(mock_embedding, test_agent):
    """Test the structure of agent metadata."""
    mock_embedding.return_value.data = [
        Mock(embedding=[0.1, 0.2, 0.3])
    ]

    router = AgentRouter()
    router.add_agent(test_agent)

    metadata = router.agent_metadata[0]
    assert "name" in metadata
    assert "text" in metadata
    assert metadata["name"] == "test_agent"
    assert (
        "test_agent A test agent You are a test agent"
        in metadata["text"]
    )


def test_agent_router_edge_cases():
    """Test various edge cases."""
    with patch(
        "swarms.structs.agent_router.embedding"
    ) as mock_embedding:
        mock_embedding.return_value.data = [
            Mock(embedding=[0.1, 0.2, 0.3])
        ]

        router = AgentRouter()

        # Test with empty string task
        result = router.find_best_agent("")
        assert result is None

        # Test with very long task description
        long_task = "test " * 1000
        result = router.find_best_agent(long_task)
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__])
