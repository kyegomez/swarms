import pytest
from types import SimpleNamespace
from swarms.structs.agent import Agent
from swarms.utils import litellm_wrapper


def fake_completion_factory(captured_requests):
    def fake_completion(**kwargs):
        captured_requests.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="response text", tool_calls=None
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                prompt_tokens_details=SimpleNamespace(cached_tokens=0),
                completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
            ),
        )
    return fake_completion


def test_update_system_prompt_reaches_llm_request(monkeypatch):
    """When update_system_prompt is called after agent creation, requests must carry the new prompt."""
    captured = []
    monkeypatch.setattr(litellm_wrapper, "completion", fake_completion_factory(captured))

    agent = Agent(agent_name="t", model_name="gpt-5.4-mini", print_on=False)
    agent.update_system_prompt("NEW-PROMPT-XYZ")
    agent.run("hi")

    assert len(captured) == 1
    messages = captured[0].get("messages", [])
    system_messages = [m for m in messages if m.get("role") == "system"]
    assert any("NEW-PROMPT-XYZ" in m.get("content", "") for m in system_messages)


def test_system_prompt_property_assignment_reaches_llm_request(monkeypatch):
    """Assigning agent.system_prompt directly must also reach the LLM request."""
    captured = []
    monkeypatch.setattr(litellm_wrapper, "completion", fake_completion_factory(captured))

    agent = Agent(agent_name="t", model_name="gpt-5.4-mini", print_on=False)
    agent.system_prompt = "DIRECT-SYSTEM-PROMPT-123"
    agent.run("hi")

    assert len(captured) == 1
    messages = captured[0].get("messages", [])
    system_messages = [m for m in messages if m.get("role") == "system"]
    assert any("DIRECT-SYSTEM-PROMPT-123" in m.get("content", "") for m in system_messages)


def test_update_system_prompt_updates_short_memory():
    """update_system_prompt updates short_memory system prompt and conversation history."""
    agent = Agent(agent_name="t", model_name="gpt-5.4-mini", system_prompt="INITIAL", print_on=False)
    assert agent.short_memory.system_prompt == "INITIAL"

    agent.update_system_prompt("UPDATED-PROMPT")
    assert agent.short_memory.system_prompt == "UPDATED-PROMPT"
    history_str = agent.short_memory.return_history_as_string()
    assert "UPDATED-PROMPT" in history_str
    assert "INITIAL" not in history_str
