from unittest.mock import Mock, patch
import pytest

from transformers import AutoModelForCausalLM, AutoTokenizer

from swarms import ToolAgent
from swarms.agents.exceptions import (
    ToolExecutionError,
    ToolNotFoundError,
    ToolParameterError,
)


def test_tool_agent_init():
    model = Mock(spec=AutoModelForCausalLM)
    tokenizer = Mock(spec=AutoTokenizer)
    json_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
            "is_student": {"type": "boolean"},
            "courses": {"type": "array", "items": {"type": "string"}},
        },
    }
    name = "Test Agent"
    description = "This is a test agent"

    agent = ToolAgent(
        name, description, model, tokenizer, json_schema
    )

    assert agent.name == name
    assert agent.description == description
    assert agent.model == model
    assert agent.tokenizer == tokenizer
    assert agent.json_schema == json_schema


@patch.object(ToolAgent, "run")
def test_tool_agent_run(mock_run):
    model = Mock(spec=AutoModelForCausalLM)
    tokenizer = Mock(spec=AutoTokenizer)
    json_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
            "is_student": {"type": "boolean"},
            "courses": {"type": "array", "items": {"type": "string"}},
        },
    }
    name = "Test Agent"
    description = "This is a test agent"
    task = (
        "Generate a person's information based on the following"
        " schema:"
    )

    agent = ToolAgent(
        name, description, model, tokenizer, json_schema
    )
    agent.run(task)

    mock_run.assert_called_once_with(task)


def test_tool_agent_init_with_kwargs():
    model = Mock(spec=AutoModelForCausalLM)
    tokenizer = Mock(spec=AutoTokenizer)
    json_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
            "is_student": {"type": "boolean"},
            "courses": {"type": "array", "items": {"type": "string"}},
        },
    }
    name = "Test Agent"
    description = "This is a test agent"

    kwargs = {
        "debug": True,
        "max_array_length": 20,
        "max_number_tokens": 12,
        "temperature": 0.5,
        "max_string_token_length": 20,
    }

    agent = ToolAgent(
        name, description, model, tokenizer, json_schema, **kwargs
    )

    assert agent.name == name
    assert agent.description == description
    assert agent.model == model
    assert agent.tokenizer == tokenizer
    assert agent.json_schema == json_schema
    assert agent.debug == kwargs["debug"]
    assert agent.max_array_length == kwargs["max_array_length"]
    assert agent.max_number_tokens == kwargs["max_number_tokens"]
    assert agent.temperature == kwargs["temperature"]
    assert (
        agent.max_string_token_length
        == kwargs["max_string_token_length"]
    )


def test_tool_agent_initialization():
    """Test tool agent initialization with valid parameters."""
    agent = ToolAgent(
        model_name="test-model", temperature=0.7, max_tokens=1000
    )
    assert agent.model_name == "test-model"
    assert agent.temperature == 0.7
    assert agent.max_tokens == 1000
    assert agent.retry_attempts == 3
    assert agent.retry_interval == 1.0


def test_tool_agent_initialization_error():
    """Test tool agent initialization with invalid model."""
    with pytest.raises(ToolExecutionError) as exc_info:
        ToolAgent(model_name="invalid-model")
    assert "model_initialization" in str(exc_info.value)


def test_tool_validation():
    """Test tool parameter validation."""
    tools_list = [
        {
            "name": "test_tool",
            "parameters": [
                {"name": "required_param", "required": True},
                {"name": "optional_param", "required": False},
            ],
        }
    ]

    agent = ToolAgent(tools_list_dictionary=tools_list)

    # Test missing required parameter
    with pytest.raises(ToolParameterError) as exc_info:
        agent._validate_tool("test_tool", {})
    assert "Missing required parameters" in str(exc_info.value)

    # Test valid parameters
    agent._validate_tool("test_tool", {"required_param": "value"})

    # Test non-existent tool
    with pytest.raises(ToolNotFoundError) as exc_info:
        agent._validate_tool("non_existent_tool", {})
    assert "Tool 'non_existent_tool' not found" in str(exc_info.value)


def test_retry_mechanism():
    """Test retry mechanism for failed operations."""
    mock_llm = Mock()
    mock_llm.generate.side_effect = [
        Exception("First attempt failed"),
        Exception("Second attempt failed"),
        Mock(outputs=[Mock(text="Success")]),
    ]

    agent = ToolAgent(model_name="test-model")
    agent.llm = mock_llm

    # Test successful retry
    result = agent.run("test task")
    assert result == "Success"
    assert mock_llm.generate.call_count == 3

    # Test all retries failing
    mock_llm.generate.side_effect = Exception("All attempts failed")
    with pytest.raises(ToolExecutionError) as exc_info:
        agent.run("test task")
    assert "All attempts failed" in str(exc_info.value)


def test_batched_execution():
    """Test batched execution with error handling."""
    mock_llm = Mock()
    mock_llm.generate.side_effect = [
        Mock(outputs=[Mock(text="Success 1")]),
        Exception("Task 2 failed"),
        Mock(outputs=[Mock(text="Success 3")]),
    ]

    agent = ToolAgent(model_name="test-model")
    agent.llm = mock_llm

    tasks = ["Task 1", "Task 2", "Task 3"]
    results = agent.batched_run(tasks)

    assert len(results) == 3
    assert results[0] == "Success 1"
    assert "Error" in results[1]
    assert results[2] == "Success 3"


def test_prompt_preparation():
    """Test prompt preparation with and without system prompt."""
    # Test without system prompt
    agent = ToolAgent()
    prompt = agent._prepare_prompt("test task")
    assert prompt == "User: test task\nAssistant:"

    # Test with system prompt
    agent = ToolAgent(system_prompt="You are a helpful assistant")
    prompt = agent._prepare_prompt("test task")
    assert (
        prompt
        == "You are a helpful assistant\n\nUser: test task\nAssistant:"
    )


def test_tool_execution_error_handling():
    """Test error handling during tool execution."""
    agent = ToolAgent(model_name="test-model")
    agent.llm = None  # Simulate uninitialized LLM

    with pytest.raises(ToolExecutionError) as exc_info:
        agent.run("test task")
    assert "LLM not initialized" in str(exc_info.value)

    # Test with invalid parameters
    with pytest.raises(ToolExecutionError) as exc_info:
        agent.run("test task", invalid_param="value")
    assert "Error running task" in str(exc_info.value)
