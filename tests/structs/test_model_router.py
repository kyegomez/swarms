import json

from swarms.structs.model_router import (
    LiteLLM,
    ModelOutput,
    ModelRouter,
)


def test_run_executes_the_task_on_the_model_the_router_picked(
    monkeypatch,
):
    calls = []

    def fake_run(self, task):
        calls.append((self.model_name, task))
        if self.response_format is ModelOutput:
            return json.dumps(
                {
                    "rationale": "short creative task",
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "task": "Write a haiku about rain",
                    "max_tokens": 100,
                    "temperature": 0.7,
                    "system_prompt": "You are a poet.",
                }
            )
        return "rain on the window"

    monkeypatch.setattr(LiteLLM, "run", fake_run)

    assert ModelRouter().run("write a haiku") == "rain on the window"
    assert calls[-1] == (
        "openai/gpt-4o-mini",
        "Write a haiku about rain",
    )
