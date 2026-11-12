import swarms.prompts.autonomous_agent_prompt as prompt_module


def _freeze(monkeypatch, value):
    monkeypatch.setattr(
        prompt_module,
        "get_time",
        lambda: f"Current date and time: {value}\n",
    )


def test_prompt_reflects_the_clock_at_call_time(monkeypatch):
    _freeze(monkeypatch, "FIRST")
    first = prompt_module.get_autonomous_agent_prompt()
    _freeze(monkeypatch, "SECOND")
    second = prompt_module.get_autonomous_agent_prompt()

    assert "Time: Current date and time: FIRST" in first
    assert "Time: Current date and time: SECOND" in second


def test_contextual_prompt_uses_the_same_clock(monkeypatch):
    _freeze(monkeypatch, "NOW")
    built = prompt_module.get_autonomous_agent_prompt_with_context(
        agent_name="Scout"
    )

    assert "Time: Current date and time: NOW" in built
    assert "You are Scout, an elite autonomous agent" in built
