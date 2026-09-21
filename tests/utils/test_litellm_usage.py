"""Provider-reported token usage on ``LiteLLM`` — offline, ``completion`` is stubbed."""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from swarms.utils.litellm_wrapper import (
    LiteLLM,
    empty_usage,
    usage_from_response,
)


def _response(prompt=10, completion=5, cached=0, total=None):
    """The shape litellm returns: choices[0].message.content plus a usage block."""
    details = (
        SimpleNamespace(cached_tokens=cached) if cached else None
    )
    usage = SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion if total is None else total,
        prompt_tokens_details=details,
    )
    message = SimpleNamespace(content="ok", tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)], usage=usage
    )


class TestUsageFromResponse:
    def test_reads_the_openai_shape(self):
        assert usage_from_response(_response(10, 5, cached=3)) == {
            "input_tokens": 10,
            "output_tokens": 5,
            "cached_tokens": 3,
            "reasoning_tokens": 0,
            "total_tokens": 15,
        }

    def test_reads_a_dict_response(self):
        response = {
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 2,
                "total_tokens": 9,
                "prompt_tokens_details": {"cached_tokens": 4},
            }
        }
        assert usage_from_response(response)["cached_tokens"] == 4

    def test_reads_reasoning_tokens_from_completion_details(self):
        response = _response(10, 68)
        response.usage.completion_tokens_details = SimpleNamespace(
            reasoning_tokens=53
        )
        usage = usage_from_response(response)
        # A breakdown of output_tokens, not an addition to it
        assert usage["output_tokens"] == 68
        assert usage["reasoning_tokens"] == 53

    def test_no_reasoning_details_means_zero_reasoning(self):
        assert (
            usage_from_response(_response())["reasoning_tokens"] == 0
        )

    def test_no_cache_details_means_zero_cached(self):
        assert usage_from_response(_response())["cached_tokens"] == 0

    def test_missing_usage_block_is_none(self):
        assert (
            usage_from_response(SimpleNamespace(choices=[])) is None
        )
        assert usage_from_response({"choices": []}) is None

    def test_total_falls_back_to_the_sum(self):
        assert (
            usage_from_response(_response(10, 5, total=0))[
                "total_tokens"
            ]
            == 15
        )


class TestLiteLLMUsage:
    def test_starts_at_zero(self):
        assert LiteLLM(model_name="gpt-5.4").usage == empty_usage()

    def test_accumulates_across_calls(self):
        llm = LiteLLM(model_name="gpt-5.4")
        with patch(
            "swarms.utils.litellm_wrapper.completion",
            side_effect=[
                _response(10, 5, cached=2),
                _response(20, 1),
            ],
        ):
            llm.run("one")
            llm.run("two")

        assert llm.usage == {
            "input_tokens": 30,
            "output_tokens": 6,
            "cached_tokens": 2,
            "reasoning_tokens": 0,
            "total_tokens": 36,
        }

    def test_hook_receives_each_call(self):
        seen = []
        llm = LiteLLM(model_name="gpt-5.4", usage_hook=seen.append)
        with patch(
            "swarms.utils.litellm_wrapper.completion",
            return_value=_response(3, 4),
        ):
            llm.run("x")

        assert seen == [
            {
                "input_tokens": 3,
                "output_tokens": 4,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
                "total_tokens": 7,
            }
        ]

    def test_arun_records_too(self):
        llm = LiteLLM(model_name="gpt-5.4")

        async def _fake(**kwargs):
            return _response(8, 2)

        with patch(
            "swarms.utils.litellm_wrapper.acompletion",
            side_effect=_fake,
        ):
            asyncio.run(llm.arun("x"))

        assert llm.usage["total_tokens"] == 10

    def test_a_response_without_usage_is_skipped(self):
        llm = LiteLLM(model_name="gpt-5.4")
        bare = _response()
        bare.usage = None
        with patch(
            "swarms.utils.litellm_wrapper.completion",
            return_value=bare,
        ):
            llm.run("x")

        assert llm.usage == empty_usage()

    @staticmethod
    def _content_chunk(text):
        delta = SimpleNamespace(content=text)
        return SimpleNamespace(
            choices=[SimpleNamespace(delta=delta)], usage=None
        )

    @staticmethod
    def _usage_chunk(prompt, completion, cached=0):
        """The trailing include_usage chunk: a usage block and no choices."""
        details = (
            SimpleNamespace(cached_tokens=cached) if cached else None
        )
        return SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=prompt + completion,
                prompt_tokens_details=details,
            ),
        )

    def test_streaming_is_counted_from_the_final_chunk(self):
        seen = []
        llm = LiteLLM(
            model_name="gpt-5.4", stream=True, usage_hook=seen.append
        )
        with patch(
            "swarms.utils.litellm_wrapper.completion",
            return_value=iter(
                [
                    self._content_chunk("hel"),
                    self._content_chunk("lo"),
                    self._usage_chunk(10, 5, cached=2),
                ]
            ),
        ):
            forwarded = [
                c.choices[0].delta.content for c in llm.run("x")
            ]

        # The usage-only chunk is consumed, never handed to the caller
        assert forwarded == ["hel", "lo"]
        assert llm.usage == {
            "input_tokens": 10,
            "output_tokens": 5,
            "cached_tokens": 2,
            "reasoning_tokens": 0,
            "total_tokens": 15,
        }
        assert seen == [llm.usage]

    def test_streaming_asks_the_provider_for_usage(self):
        llm = LiteLLM(model_name="gpt-5.4", stream=True)
        with patch(
            "swarms.utils.litellm_wrapper.completion",
            return_value=iter([]),
        ) as fake:
            list(llm.run("x"))

        assert fake.call_args.kwargs["stream_options"] == {
            "include_usage": True
        }

    def test_a_stream_without_a_usage_chunk_leaves_usage_unchanged(
        self,
    ):
        llm = LiteLLM(model_name="gpt-5.4", stream=True)
        with patch(
            "swarms.utils.litellm_wrapper.completion",
            return_value=iter([self._content_chunk("hi")]),
        ):
            list(llm.run("x"))

        assert llm.usage == empty_usage()

    def test_non_streaming_requests_do_not_set_stream_options(self):
        llm = LiteLLM(model_name="gpt-5.4")
        with patch(
            "swarms.utils.litellm_wrapper.completion",
            return_value=_response(1, 1),
        ) as fake:
            llm.run("x")

        assert "stream_options" not in fake.call_args.kwargs
