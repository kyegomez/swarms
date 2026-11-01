import swarms.prompts.autonomous_agent_prompt as prompt_module
from swarms.structs.autonomous_loop_utils import get_planning_prompt


def _freeze(monkeypatch, value):
    monkeypatch.setattr(
        prompt_module,
        "get_time",
        lambda: f"Current date and time: {value}\n",
    )


def test_clock_is_in_the_planning_turn_not_the_system_prompt(
    monkeypatch,
):
    _freeze(monkeypatch, "FIRST")
    first = prompt_module.get_autonomous_agent_prompt()
    _freeze(monkeypatch, "SECOND")
    second = prompt_module.get_autonomous_agent_prompt()
    planning = get_planning_prompt("Ship it")

    assert first == second
    assert "Current date and time" not in first
    assert planning.startswith("Current date and time: ")
    assert "Ship it" in planning


def test_contextual_prompt_names_the_agent():
    built = prompt_module.get_autonomous_agent_prompt_with_context(
        agent_name="Scout"
    )

    assert "You are Scout, an elite autonomous agent" in built
