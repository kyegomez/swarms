import json
from types import SimpleNamespace

from swarms.structs.auto_agent_builder import AutoAgentBuilder
from swarms.utils import litellm_wrapper


def test_usage_adds_up_every_build_configs_call(monkeypatch):
    """builder.usage keeps a running total of the builder agent's provider usage."""
    roster = {
        "agents": [
            {
                "name": "Researcher",
                "description": "Finds sources.",
                "system_prompt": "You research.",
                "model_name": "gpt-5.4-mini",
            }
        ]
    }

    def fake_completion(**kwargs):
        tool_call = SimpleNamespace(
            id="call_1",
            type="function",
            function=SimpleNamespace(
                name="build_agents", arguments=json.dumps(roster)
            ),
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None, tool_calls=[tool_call]
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=10,
                total_tokens=110,
                prompt_tokens_details=SimpleNamespace(
                    cached_tokens=4
                ),
                completion_tokens_details=SimpleNamespace(
                    reasoning_tokens=3
                ),
            ),
        )

    monkeypatch.setattr(
        litellm_wrapper, "completion", fake_completion
    )

    builder = AutoAgentBuilder(return_dict=True)
    assert builder.run("Research EVs.")[0]["name"] == "Researcher"
    once = {
        "input_tokens": 100,
        "output_tokens": 10,
        "cached_tokens": 4,
        "reasoning_tokens": 3,
        "total_tokens": 110,
    }
    assert builder.usage == once

    builder.run("Research EVs again.")
    assert builder.usage == {k: 2 * v for k, v in once.items()}
