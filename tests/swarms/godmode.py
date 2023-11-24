from unittest.mock import patch
from swarms.swarms.god_mode import GodMode, LLM


def test_godmode_initialization():
    godmode = GodMode(llms=[LLM] * 5)
    assert isinstance(godmode, GodMode)
    assert len(godmode.llms) == 5


def test_godmode_run(monkeypatch):
    def mock_llm_run(self, task):
        return "response"

    monkeypatch.setattr(LLM, "run", mock_llm_run)
    godmode = GodMode(llms=[LLM] * 5)
    responses = godmode.run("task1")
    assert len(responses) == 5
    assert responses == [
        "response",
        "response",
        "response",
        "response",
        "response",
    ]


@patch("builtins.print")
def test_godmode_print_responses(mock_print, monkeypatch):
    def mock_llm_run(self, task):
        return "response"

    monkeypatch.setattr(LLM, "run", mock_llm_run)
    godmode = GodMode(llms=[LLM] * 5)
    godmode.print_responses("task1")
    assert mock_print.call_count == 1
