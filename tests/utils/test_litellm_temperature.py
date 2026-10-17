"""Regression tests for optional temperature in LiteLLM (issue #2390).

``Agent`` and ``LiteLLM`` defaulted to ``temperature=0.5`` and
``_build_completion_params`` always included it, so current Claude models
reject the very first request of a default-configured agent with
HTTP 400 "temperature is deprecated for this model".

The default is now ``None`` and ``temperature`` is only added to the
completion params when explicitly set — mirroring how ``top_p`` is handled.
"""

import pytest

from swarms.utils.litellm_wrapper import LiteLLM


def test_temperature_omitted_when_unset():
    """A default-configured wrapper must not send temperature at all."""
    llm = LiteLLM(model_name="gpt-4")
    params = llm._build_completion_params("hello")
    assert "temperature" not in params


def test_explicit_temperature_still_passed():
    """An explicitly set temperature must still reach the request."""
    llm = LiteLLM(model_name="gpt-4", temperature=0.3)
    params = llm._build_completion_params("hello")
    assert params["temperature"] == 0.3


def test_none_temperature_can_be_overridden_by_runtime_kwargs():
    """Runtime kwargs may still inject temperature for callers that want it."""
    llm = LiteLLM(model_name="gpt-4")
    params = llm._build_completion_params(
        "hello", runtime_kwargs={"temperature": 0.9}
    )
    assert params["temperature"] == 0.9
