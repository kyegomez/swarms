"""The output-token cap must reach each provider under the key it accepts.

OpenAI's reasoning-era models reject ``max_tokens`` and require
``max_completion_tokens``; everything else takes ``max_tokens``. LiteLLM does
not translate, and ``drop_params`` cannot help because LiteLLM believes both
keys are supported. Offline: ``completion`` is stubbed throughout.
"""

import asyncio
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


class TestOutputTokenKeyByModel:
    @pytest.mark.parametrize(
        "model", ["gpt-5.4", "gpt-6-astra", "o3", "openai/o4-mini"]
    )
    def test_openai_reasoning_families_get_max_completion_tokens(
        self, model
    ):
        sent = _params_sent(model)
        assert sent["max_completion_tokens"] == 1234
        assert "max_tokens" not in sent

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-4o-mini",
            "claude-sonnet-5",
            "gemini/gemini-2.5-pro",
            "groq/llama-3.3-70b-versatile",
        ],
    )
    def test_everything_else_keeps_max_tokens(self, model):
        sent = _params_sent(model)
        assert sent["max_tokens"] == 1234
        assert "max_completion_tokens" not in sent


class TestRetryOnRejectedKey:
    """A model the family list does not know is still handled: the
    provider's own rejection says which key it wants, so retry once.
    """

    REJECT_MAX_TOKENS = (
        "OpenAIException - Unsupported parameter: 'max_tokens' is not "
        "supported with this model. Use 'max_completion_tokens' instead."
    )

    def test_retries_with_max_completion_tokens_when_asked(self):
        llm = LiteLLM(model_name="gpt-4o-mini", max_tokens=99)
        calls = []

        def fake(**kwargs):
            calls.append(kwargs)
            if "max_tokens" in kwargs:
                raise RuntimeError(self.REJECT_MAX_TOKENS)
            return _response()

        with patch(
            "swarms.utils.litellm_wrapper.completion",
            side_effect=fake,
        ):
            assert llm.run("hi") == "ok"

        assert len(calls) == 2
        assert "max_tokens" in calls[0]
        assert calls[1]["max_completion_tokens"] == 99
        assert "max_tokens" not in calls[1]

    def test_retries_the_other_way_too(self):
        llm = LiteLLM(model_name="gpt-5.4", max_tokens=99)
        calls = []

        def fake(**kwargs):
            calls.append(kwargs)
            if "max_completion_tokens" in kwargs:
                raise RuntimeError(
                    "Unsupported parameter 'max_completion_tokens', "
                    "use 'max_tokens'"
                )
            return _response()

        with patch(
            "swarms.utils.litellm_wrapper.completion",
            side_effect=fake,
        ):
            assert llm.run("hi") == "ok"

        assert calls[1]["max_tokens"] == 99

    def test_unrelated_errors_are_not_retried(self):
        llm = LiteLLM(model_name="gpt-4o-mini", max_tokens=99)
        calls = []

        def fake(**kwargs):
            calls.append(kwargs)
            raise RuntimeError("invalid api key")

        with patch(
            "swarms.utils.litellm_wrapper.completion",
            side_effect=fake,
        ):
            with pytest.raises(RuntimeError, match="invalid api key"):
                llm.run("hi")

        assert len(calls) == 1

    def test_a_second_rejection_propagates(self):
        llm = LiteLLM(model_name="gpt-4o-mini", max_tokens=99)

        with patch(
            "swarms.utils.litellm_wrapper.completion",
            side_effect=RuntimeError(self.REJECT_MAX_TOKENS),
        ) as fake:
            with pytest.raises(RuntimeError):
                llm.run("hi")

        assert fake.call_count == 2

    def test_arun_retries_too(self):
        llm = LiteLLM(model_name="gpt-4o-mini", max_tokens=99)
        calls = []

        async def fake(**kwargs):
            calls.append(kwargs)
            if "max_tokens" in kwargs:
                raise RuntimeError(self.REJECT_MAX_TOKENS)
            return _response()

        with patch(
            "swarms.utils.litellm_wrapper.acompletion",
            side_effect=fake,
        ):
            assert asyncio.run(llm.arun("hi")) == "ok"

        assert calls[1]["max_completion_tokens"] == 99
