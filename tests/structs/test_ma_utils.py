from types import SimpleNamespace

from swarms.structs.agent import Agent
from swarms.structs.ma_utils import set_random_models_for_agents
from swarms.utils import litellm_wrapper


def test_agents_call_the_model_they_were_assigned(monkeypatch):
    sent = []

    def fake_completion(**params):
        sent.append(params["model"])
        message = SimpleNamespace(
            content="ok", tool_calls=None, reasoning_content=None
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=message, finish_reason="stop")
            ],
            usage=None,
        )

    monkeypatch.setattr(
        litellm_wrapper, "completion", fake_completion
    )

    agents = [
        Agent(agent_name=name, model_name="gpt-5.4", max_loops=1)
        for name in ("first", "second")
    ]
    set_random_models_for_agents(agents, model_names=["gpt-4o-mini"])
    for agent in agents:
        agent.run("hi")

    assert sent == ["gpt-4o-mini", "gpt-4o-mini"]
