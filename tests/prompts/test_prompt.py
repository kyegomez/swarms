import json
import pytest
from swarms.prompts.prompt import Prompt
from swarms.prompts import AUTONOMOUS_AGENT_SYSTEM_PROMPT, autonomous_agent_system_prompt


def dummy_tool(x: int) -> int:
    """Double the given integer."""
    return x * 2


def test_prompt_initialization_and_editing():
    prompt = Prompt(content="Initial prompt content")
    assert prompt.get_prompt() == "Initial prompt content"
    assert prompt.edit_count == 0
    assert len(prompt.edit_history) == 1

    prompt.edit_prompt("Updated prompt content")
    assert prompt.get_prompt() == "Updated prompt content"
    assert prompt.edit_count == 1
    assert len(prompt.edit_history) == 2

    # Duplicate content raises ValueError
    with pytest.raises(ValueError):
        prompt.edit_prompt("Updated prompt content")


def test_prompt_rollback():
    prompt = Prompt(content="Version 0")
    prompt.edit_prompt("Version 1")
    prompt.edit_prompt("Version 2")

    assert prompt.get_prompt() == "Version 2"
    prompt.rollback(0)
    assert prompt.get_prompt() == "Version 0"
    assert prompt.edit_count == 0


def test_prompt_add_tools():
    prompt = Prompt(content="System prompt before tools.")
    updated_content = prompt.add_tools([dummy_tool])

    assert updated_content == prompt.content
    assert "dummy_tool" in prompt.content
    assert "Double the given integer." in prompt.content
    assert prompt.edit_count == 1
    assert len(prompt.edit_history) == 2


def test_prompt_json_export():
    prompt = Prompt(name="TestPrompt", content="Prompt content here")
    data = json.loads(prompt.return_json())
    assert data["name"] == "TestPrompt"
    assert data["content"] == "Prompt content here"


def test_autonomous_agent_system_prompt_alias():
    assert AUTONOMOUS_AGENT_SYSTEM_PROMPT is autonomous_agent_system_prompt
    result = AUTONOMOUS_AGENT_SYSTEM_PROMPT()
    assert isinstance(result, str)
    assert "You are an elite autonomous agent" in result
