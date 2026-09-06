import re

import swarms.prompts.agent_system_prompts as prompt_module

TIMESTAMP = re.compile(r"Current date and time: [^\n]*")


def _freeze(monkeypatch, value):
    monkeypatch.setattr(
        prompt_module,
        "get_time",
        lambda: f"Current date and time: {value}\n",
    )


def test_constant_carries_no_timestamp():
    assert not TIMESTAMP.search(prompt_module.AGENT_SYSTEM_PROMPT_3)


def test_builder_reflects_the_clock_at_call_time(monkeypatch):
    _freeze(monkeypatch, "FIRST")
    first = prompt_module.build_agent_system_prompt()
    _freeze(monkeypatch, "SECOND")
    second = prompt_module.build_agent_system_prompt()

    assert "FIRST" in first
    assert "SECOND" in second
    assert first != second


def test_builder_is_the_constant_plus_a_time_line(monkeypatch):
    _freeze(monkeypatch, "ONLY")
    built = prompt_module.build_agent_system_prompt()

    assert (
        built.replace("Time: Current date and time: ONLY\n\n", "")
        == prompt_module.AGENT_SYSTEM_PROMPT_3
    )
