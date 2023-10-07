import pytest
from unittest.mock import Mock, patch
from swarms.agents.agents import (
    AgentNodeInitializer,
    AgentNode,
    agent,
)  # replace with actual import


# For initializing AgentNodeInitializer in multiple tests
@pytest.fixture
def mock_agent_node_initializer():
    with patch("swarms.agents.agents.ChatOpenAI") as mock_llm, patch(
        "swarms.agents.agents.AutoGPT"
    ) as mock_agent:
        initializer = AgentNodeInitializer(
            model_type="openai",
            model_id="test",
            openai_api_key="test_key",
            temperature=0.5,
        )
        initializer.llm = mock_llm
        initializer.tools = [Mock(spec=BaseTool)]
        initializer.vectorstore = Mock()
        initializer.agent = mock_agent

    return initializer


# Test initialize_llm method of AgentNodeInitializer class
@pytest.mark.parametrize("model_type", ["openai", "huggingface", "invalid"])
def test_agent_node_initializer_initialize_llm(model_type, mock_agent_node_initializer):
    with patch("swarms.agents.agents.ChatOpenAI") as mock_openai, patch(
        "swarms.agents.agents.HuggingFaceLLM"
    ) as mock_huggingface:
        if model_type == "invalid":
            with pytest.raises(ValueError):
                mock_agent_node_initializer.initialize_llm(
                    model_type, "model_id", "openai_api_key", 0.5
                )
        else:
            mock_agent_node_initializer.initialize_llm(
                model_type, "model_id", "openai_api_key", 0.5
            )
            if model_type == "openai":
                mock_openai.assert_called_once()
            elif model_type == "huggingface":
                mock_huggingface.assert_called_once()


# Test add_tool method of AgentNodeInitializer class
def test_agent_node_initializer_add_tool(mock_agent_node_initializer):
    with patch("swarms.agents.agents.BaseTool") as mock_base_tool:
        mock_agent_node_initializer.add_tool(mock_base_tool)
        assert mock_base_tool in mock_agent_node_initializer.tools


# Test run method of AgentNodeInitializer class
@pytest.mark.parametrize("prompt", ["valid prompt", ""])
def test_agent_node_initializer_run(prompt, mock_agent_node_initializer):
    if prompt == "":
        with pytest.raises(ValueError):
            mock_agent_node_initializer.run(prompt)
    else:
        assert mock_agent_node_initializer.run(prompt) == "Task completed by AgentNode"


# For initializing AgentNode in multiple tests
@pytest.fixture
def mock_agent_node():
    with patch("swarms.agents.agents.ChatOpenAI") as mock_llm, patch(
        "swarms.agents.agents.AgentNodeInitializer"
    ) as mock_agent_node_initializer:
        mock_agent_node = AgentNode("test_key")
        mock_agent_node.llm_class = mock_llm
        mock_agent_node.vectorstore = Mock()
        mock_agent_node_initializer.llm = mock_llm

    return mock_agent_node


# Test initialize_llm method of AgentNode class
@pytest.mark.parametrize("llm_class", ["openai", "huggingface"])
def test_agent_node_initialize_llm(llm_class, mock_agent_node):
    with patch("swarms.agents.agents.ChatOpenAI") as mock_openai, patch(
        "swarms.agents.agents.HuggingFaceLLM"
    ) as mock_huggingface:
        mock_agent_node.initialize_llm(llm_class)
        if llm_class == "openai":
            mock_openai.assert_called_once()
        elif llm_class == "huggingface":
            mock_huggingface.assert_called_once()


# Test initialize_tools method of AgentNode class
def test_agent_node_initialize_tools(mock_agent_node):
    with patch("swarms.agents.agents.DuckDuckGoSearchRun") as mock_ddg, patch(
        "swarms.agents.agents.WriteFileTool"
    ) as mock_write_file, patch(
        "swarms.agents.agents.ReadFileTool"
    ) as mock_read_file, patch(
        "swarms.agents.agents.process_csv"
    ) as mock_process_csv, patch(
        "swarms.agents.agents.WebpageQATool"
    ) as mock_webpage_qa:
        mock_agent_node.initialize_tools("openai")
        assert mock_ddg.called
        assert mock_write_file.called
        assert mock_read_file.called
        assert mock_process_csv.called
        assert mock_webpage_qa.called


# Test create_agent method of AgentNode class
def test_agent_node_create_agent(mock_agent_node):
    with patch.object(mock_agent_node, "initialize_llm"), patch.object(
        mock_agent_node, "initialize_tools"
    ), patch.object(mock_agent_node, "initialize_vectorstore"), patch(
        "swarms.agents.agents.AgentNodeInitializer"
    ) as mock_agent_node_initializer:
        mock_agent_node.create_agent()
        mock_agent_node_initializer.assert_called_once()
        mock_agent_node_initializer.return_value.create_agent.assert_called_once()


# Test agent function
@pytest.mark.parametrize(
    "openai_api_key,objective",
    [("valid_key", "valid_objective"), ("", "valid_objective"), ("valid_key", "")],
)
def test_agent(openai_api_key, objective):
    if openai_api_key == "" or objective == "":
        with pytest.raises(ValueError):
            agent(openai_api_key, objective)
    else:
        with patch(
            "swarms.agents.agents.AgentNodeInitializer"
        ) as mock_agent_node_initializer:
            mock_agent_node = (
                mock_agent_node_initializer.return_value.create_agent.return_value
            )
            mock_agent_node.run.return_value = "Agent output"
            result = agent(openai_api_key, objective)
            assert result == "Agent output"
