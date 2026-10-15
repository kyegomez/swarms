import asyncio
import json
import threading

import litellm

from swarms.structs.model_router import (
    LiteLLM,
    ModelOutput,
    ModelRouter,
    model_recommendations,
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


def test_async_run_returns_the_output_without_blocking_the_loop(
    monkeypatch,
):
    both_running = threading.Barrier(2, timeout=5)

    def fake_run(self, task):
        if self.response_format is ModelOutput:
            return json.dumps(
                {
                    "rationale": "short task",
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "task": task,
                    "max_tokens": 100,
                    "temperature": 0.7,
                    "system_prompt": "Answer briefly.",
                }
            )
        both_running.wait()
        return f"answer to {task}"

    monkeypatch.setattr(LiteLLM, "run", fake_run)
    router = ModelRouter()

    async def run_both():
        return await asyncio.gather(
            router.async_run("a"), router.async_run("b")
        )

    assert asyncio.run(run_both()) == ["answer to a", "answer to b"]


def test_every_recommended_provider_is_a_litellm_prefix():
    for model, entry in model_recommendations.items():
        assert entry["provider"] in litellm.provider_list, model
