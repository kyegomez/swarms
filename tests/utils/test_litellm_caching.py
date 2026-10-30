"""Regression tests for deprecation of the caching parameter in LiteLLM (issue #2864).

LiteLLM(caching=True) and Agent(llm_args={"caching": True}) enabled litellm's
full-response cache. RouteHub accepts the keyword for compatibility and ignores it.
The caching parameter is deprecated, emits a DeprecationWarning when enabled,
and is no longer forwarded in _build_completion_params.
"""

import warnings
import pytest

from swarms.utils.litellm_wrapper import LiteLLM
from swarms.structs.agent import Agent


def test_caching_default_emits_no_warning():
    """Default LiteLLM initialization must not emit a DeprecationWarning."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        llm = LiteLLM(model_name="gpt-4o-mini")
        assert llm.caching is False
        deprecation_warnings = [
            w
            for w in recorded
            if issubclass(w.category, DeprecationWarning)
            and "caching" in str(w.message).lower()
        ]
        assert len(deprecation_warnings) == 0


def test_caching_false_emits_no_warning():
    """Explicitly passing caching=False must not emit a DeprecationWarning."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        llm = LiteLLM(model_name="gpt-4o-mini", caching=False)
        assert llm.caching is False
        deprecation_warnings = [
            w
            for w in recorded
            if issubclass(w.category, DeprecationWarning)
            and "caching" in str(w.message).lower()
        ]
        assert len(deprecation_warnings) == 0


def test_caching_true_emits_deprecation_warning():
    """Passing caching=True must emit a DeprecationWarning."""
    with pytest.deprecated_call(match=r"(?i)caching.*deprecated"):
        llm = LiteLLM(model_name="gpt-4o-mini", caching=True)
        assert llm.caching is True


def test_caching_not_in_completion_params():
    """_build_completion_params must not include 'caching' in the request dictionary."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        llm_with_cache = LiteLLM(model_name="gpt-4o-mini", caching=True)
        params_with = llm_with_cache._build_completion_params("hello")
        assert "caching" not in params_with

    llm_without_cache = LiteLLM(model_name="gpt-4o-mini", caching=False)
    params_without = llm_without_cache._build_completion_params("hello")
    assert "caching" not in params_without


def test_agent_with_caching_in_llm_args_emits_deprecation_warning():
    """Constructing an Agent with caching=True in llm_args must emit DeprecationWarning."""
    with pytest.deprecated_call(match=r"(?i)caching.*deprecated"):
        Agent(
            agent_name="CacheTestAgent",
            model_name="gpt-4o-mini",
            llm_args={"caching": True},
            max_loops=1,
            print_on=False,
            persistent_memory=False,
        )
