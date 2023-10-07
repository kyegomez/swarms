import pytest
import logging
from unittest.mock import patch
from swarms.swarms.swarms import (
    HierarchicalSwarm,
)  # replace with your actual module name


@pytest.fixture
def swarm():
    return HierarchicalSwarm(
        model_id="gpt-4",
        openai_api_key="some_api_key",
        use_vectorstore=True,
        embedding_size=1024,
        use_async=False,
        human_in_the_loop=True,
        model_type="openai",
        boss_prompt="boss",
        worker_prompt="worker",
        temperature=0.5,
        max_iterations=100,
        logging_enabled=True,
    )


@pytest.fixture
def swarm_no_logging():
    return HierarchicalSwarm(logging_enabled=False)


def test_swarm_init(swarm):
    assert swarm.model_id == "gpt-4"
    assert swarm.openai_api_key == "some_api_key"
    assert swarm.use_vectorstore
    assert swarm.embedding_size == 1024
    assert not swarm.use_async
    assert swarm.human_in_the_loop
    assert swarm.model_type == "openai"
    assert swarm.boss_prompt == "boss"
    assert swarm.worker_prompt == "worker"
    assert swarm.temperature == 0.5
    assert swarm.max_iterations == 100
    assert swarm.logging_enabled
    assert isinstance(swarm.logger, logging.Logger)


def test_swarm_no_logging_init(swarm_no_logging):
    assert not swarm_no_logging.logging_enabled
    assert swarm_no_logging.logger.disabled


@patch("your_module.OpenAI")
@patch("your_module.HuggingFaceLLM")
def test_initialize_llm(mock_huggingface, mock_openai, swarm):
    swarm.initialize_llm("openai")
    mock_openai.assert_called_once_with(openai_api_key="some_api_key", temperature=0.5)

    swarm.initialize_llm("huggingface")
    mock_huggingface.assert_called_once_with(model_id="gpt-4", temperature=0.5)


@patch("your_module.HierarchicalSwarm.initialize_llm")
def test_initialize_tools(mock_llm, swarm):
    mock_llm.return_value = "mock_llm_class"
    tools = swarm.initialize_tools("openai")
    assert "mock_llm_class" in tools


@patch("your_module.HierarchicalSwarm.initialize_llm")
def test_initialize_tools_with_extra_tools(mock_llm, swarm):
    mock_llm.return_value = "mock_llm_class"
    tools = swarm.initialize_tools("openai", extra_tools=["tool1", "tool2"])
    assert "tool1" in tools
    assert "tool2" in tools


@patch("your_module.OpenAIEmbeddings")
@patch("your_module.FAISS")
def test_initialize_vectorstore(mock_faiss, mock_openai_embeddings, swarm):
    mock_openai_embeddings.return_value.embed_query = "embed_query"
    swarm.initialize_vectorstore()
    mock_faiss.assert_called_once_with(
        "embed_query", instance_of(faiss.IndexFlatL2), instance_of(InMemoryDocstore), {}
    )
