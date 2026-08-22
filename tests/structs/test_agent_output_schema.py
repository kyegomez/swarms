"""Tests for Agent-level structured output (``output_schema``).

Covers:
- ``response_format`` threading into the built ``LiteLLM`` instance
- validation of valid JSON responses into the Pydantic model
- retry-on-schema-mismatch inside the existing retry loop
- retry exhaustion returning None
- accepting a model instance as well as a model class
- JSON (not repr) storage in conversation memory
- fail-fast rejection of non-Pydantic schemas
- unchanged default behaviour when ``output_schema`` is unset
"""

import pytest
from pydantic import BaseModel

from swarms import Agent


class Weather(BaseModel):
    city: str
    temperature: float


class FakeLLM:
    """Network-free stand-in for LiteLLM with scripted replies.

    ``replies`` is consumed in order; the last reply repeats once the list
    is exhausted, so ``replies=["invalid"]`` means "always invalid".
    """

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.stream = False
        self.temperature = 0.5

    def run(self, task=None, img=None, **kwargs):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply


def make_agent(**kwargs):
    settings = dict(
        agent_name="schema-agent",
        model_name="gpt-4o-mini",
        max_loops=1,
        persistent_memory=False,
        print_on=False,
        verbose=False,
    )
    settings.update(kwargs)
    agent = Agent(**settings)
    return agent


VALID = '{"city": "Paris", "temperature": 18.5}'
INVALID = '{"city": "Paris"}'  # missing required field "temperature"


def test_output_schema_threaded_to_llm():
    agent = make_agent(output_schema=Weather)
    assert agent.llm.response_format is Weather


def test_output_schema_valid_response():
    agent = make_agent(output_schema=Weather, retry_attempts=2)
    agent.llm = FakeLLM([VALID])

    result = agent.run("What is the weather in Paris?")

    assert isinstance(result, Weather)
    assert result.city == "Paris"
    assert result.temperature == 18.5


def test_output_schema_retries_on_invalid_then_succeeds():
    agent = make_agent(output_schema=Weather, retry_attempts=2)
    llm = FakeLLM([INVALID, VALID])
    agent.llm = llm

    result = agent.run("What is the weather in Paris?")

    assert isinstance(result, Weather)
    assert result.city == "Paris"
    assert llm.calls == 2  # invalid first, valid second


def test_output_schema_exhausts_retries():
    agent = make_agent(output_schema=Weather, retry_attempts=2)
    llm = FakeLLM([INVALID])
    agent.llm = llm

    result = agent.run("What is the weather in Paris?")

    assert result is None
    assert llm.calls == 2  # every retry attempt was consumed


def test_output_schema_accepts_model_instance():
    agent = make_agent(
        output_schema=Weather(city="Paris", temperature=18.5),
        retry_attempts=2,
    )
    agent.llm = FakeLLM(['{"city": "Lyon", "temperature": 21.0}'])

    result = agent.run("What is the weather in Lyon?")

    assert isinstance(result, Weather)
    assert result.city == "Lyon"
    assert result.temperature == 21.0


def test_output_schema_memory_stores_json_not_repr():
    agent = make_agent(output_schema=Weather, retry_attempts=2)
    agent.llm = FakeLLM([VALID])

    agent.run("What is the weather in Paris?")

    history = agent.short_memory.return_history_as_string()
    assert '"city":"Paris"' in history  # model_dump_json() is compact
    assert "city='Paris'" not in history  # pydantic repr would leak


def test_output_schema_rejects_non_pydantic():
    with pytest.raises(ValueError, match="output_schema"):
        make_agent(output_schema=dict)


def test_no_output_schema_keeps_default_behavior():
    agent = make_agent()
    agent.llm = FakeLLM(["plain reply"])

    result = agent.run("Say hi")

    assert result == "plain reply"