"""The output-token cap is sent as max_tokens for every model.

RouteHub renames it to max_completion_tokens for providers that require
that key, such as OpenAI's reasoning models, so the client never guesses.
Offline: completion is stubbed throughout.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from swarms.utils.litellm_wrapper import LiteLLM


def _response():
    message = SimpleNamespace(content="ok", tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)], usage=None
    )


def _params_sent(model_name):
    llm = LiteLLM(model_name=model_name, max_tokens=1234)
    with patch(
        "swarms.utils.litellm_wrapper.completion",
        return_value=_response(),
    ) as fake:
        llm.run("hi")
    return fake.call_args.kwargs


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5.4",
        "o3",
        "openai/o4-mini",
        "gpt-4o-mini",
        "claude-sonnet-5",
        "gemini/gemini-2.5-pro",
        "groq/llama-3.3-70b-versatile",
    ],
)
def test_every_model_gets_max_tokens(model):
    sent = _params_sent(model)
    assert sent["max_tokens"] == 1234
    assert "max_completion_tokens" not in sent


def test_a_rejected_request_is_not_retried():
    llm = LiteLLM(model_name="gpt-4o-mini", max_tokens=99)

    with patch(
        "swarms.utils.litellm_wrapper.completion",
        side_effect=RuntimeError("invalid api key"),
    ) as fake:
        with pytest.raises(RuntimeError, match="invalid api key"):
            llm.run("hi")

    assert fake.call_count == 1
