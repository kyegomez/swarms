import asyncio
from types import SimpleNamespace

import litellm
import pytest

from swarms import (
    Agent,
    HeavySwarm,
    HierarchicalSwarm,
    PlannerWorkerSwarm,
)
from swarms.structs.auto_agent_builder import AutoAgentBuilder
from swarms.structs.groupchat import RESPOND_TOOL, GroupChat
from swarms.tools.tool_choice import forced_tool_choice
from swarms.utils import llm_backend
from swarms.utils.litellm_wrapper import LiteLLM

SCHEMA = {"type": "function", "function": {"name": "respond"}}
FORCED = {"type": "function", "function": {"name": "respond"}}
NO_FORCE = "anthropic/test-model-without-forced-tools"


@pytest.fixture(autouse=True)
def listed_models(monkeypatch):
    """Models in LiteLLM's model list, including an entry that does not say either way."""
    monkeypatch.setitem(
        litellm.model_cost,
        "test-model-without-forced-tools",
        {"supports_forced_tool_use": False},
    )
    monkeypatch.setitem(
        litellm.model_cost,
        "test-model-with-forced-tools",
        {"supports_forced_tool_use": True},
    )
    monkeypatch.setitem(
        litellm.model_cost,
        "reseller/test-model-without-forced-tools",
        {"max_tokens": 1000},
    )


@pytest.mark.parametrize(
    "tool, settings, expected",
    [
        (SCHEMA, {}, FORCED),
        ("respond", {}, FORCED),
        (SCHEMA, {"reasoning_effort": "low"}, "auto"),
        (SCHEMA, {"reasoning_enabled": True}, "auto"),
        (SCHEMA, {"reasoning_enabled": False}, FORCED),
        (SCHEMA, {"model_name": NO_FORCE}, "auto"),
        (
            SCHEMA,
            {"model_name": "test-model-without-forced-tools"},
            "auto",
        ),
        (
            SCHEMA,
            {"model_name": "test-model-with-forced-tools"},
            FORCED,
        ),
        (SCHEMA, {"model_name": "unlisted-model"}, FORCED),
        (
            SCHEMA,
            {
                "model_name": "reseller/test-model-without-forced-tools"
            },
            "auto",
        ),
    ],
)
def test_forced_tool_choice_names_the_tool_unless_reasoning_is_on_or_the_model_cannot_be_forced(
    tool, settings, expected
):
    assert forced_tool_choice(tool, **settings) == expected


def _worker():
    return Agent(
        agent_name="Worker",
        model_name="gpt-4o-mini",
        max_loops=1,
        print_on=False,
    )


def _planner_request():
    swarm = PlannerWorkerSwarm(
        agents=[_worker()], planner_model_name=NO_FORCE
    )
    swarm._create_planner_agent().llm.run(task="plan")


def _director_request():
    swarm = HierarchicalSwarm(
        agents=[_worker()],
        director_model_name=NO_FORCE,
        autosave=False,
    )
    swarm.director.llm.run(task="plan")


def _question_request():
    swarm = HeavySwarm(
        worker_model_name="gpt-4o-mini",
        question_agent_model_name=NO_FORCE,
    )
    swarm.execute_question_generation("A task")


def _builder_request():
    AutoAgentBuilder(model_name=NO_FORCE)._builder_agent().llm.run(
        task="build"
    )


def _bid_request():
    agents = [
        Agent(
            agent_name=name,
            model_name=NO_FORCE,
            max_loops=1,
            print_on=False,
            tools_list_dictionary=[RESPOND_TOOL],
        )
        for name in ("A", "B")
    ]
    GroupChat(agents=agents).run("topic")


@pytest.mark.parametrize(
    "request_tool_call",
    [
        _planner_request,
        _director_request,
        _question_request,
        _builder_request,
        _bid_request,
    ],
    ids=[
        "planner",
        "director",
        "questions",
        "builder",
        "groupchat_bid",
    ],
)
def test_every_forced_call_sends_auto_to_a_model_that_cannot_be_forced(
    monkeypatch, request_tool_call
):
    """Claude Sonnet 5.5 answers a forced tool_choice with a 400."""
    sent = []

    def capture(**params):
        sent.append(params)
        raise RuntimeError("captured")

    monkeypatch.setattr(
        "swarms.utils.litellm_wrapper.completion", capture
    )

    try:
        request_tool_call()
    except Exception:
        pass

    assert sent and sent[0]["tool_choice"] == "auto"


def test_the_routehub_backend_never_reads_litellm(monkeypatch):
    """Reading the list on RouteHub loaded LiteLLM and fetched its model registry."""

    def no_litellm():
        raise AssertionError("LiteLLM was loaded")

    monkeypatch.setattr(llm_backend, "BACKEND", "routehub")
    monkeypatch.setattr(llm_backend, "backend_module", no_litellm)

    assert forced_tool_choice(SCHEMA, model_name=NO_FORCE) == FORCED


UNSUPPORTED = (
    'AnthropicException - tool_choice: type "tool" and "any" '
    "are not supported for this model."
)


@pytest.mark.parametrize("entry", ["run", "arun"])
@pytest.mark.parametrize(
    "error, expected_sent",
    [
        (UNSUPPORTED, [FORCED, "auto"]),
        (
            "Invalid tool_choice: function 'respond' was not found in tools",
            [FORCED],
        ),
        (
            "tool_choice must be one of none, auto, required, or a function",
            [FORCED],
        ),
    ],
    ids=[
        "unsupported",
        "unknown_tool_name",
        "other_tool_choice_error",
    ],
)
def test_only_an_unsupported_forced_choice_is_retried_with_auto(
    monkeypatch, entry, error, expected_sent
):
    """A model the list does not cover still gets its call; any other tool_choice error still raises."""
    sent = []

    def completion(**params):
        sent.append(params["tool_choice"])
        if isinstance(params["tool_choice"], dict):
            raise RuntimeError(error)
        message = SimpleNamespace(content="ok", tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)], usage=None
        )

    async def acompletion(**params):
        return completion(**params)

    monkeypatch.setattr(
        "swarms.utils.litellm_wrapper.completion", completion
    )
    monkeypatch.setattr(
        "swarms.utils.litellm_wrapper.acompletion", acompletion
    )
    llm = LiteLLM(
        model_name="anthropic/unlisted-claude-model",
        tools_list_dictionary=[SCHEMA],
        tool_choice=forced_tool_choice(
            SCHEMA, model_name="anthropic/unlisted-claude-model"
        ),
    )

    try:
        result = getattr(llm, entry)("hi")
        if asyncio.iscoroutine(result):
            asyncio.run(result)
    except RuntimeError:
        assert len(expected_sent) == 1

    assert sent == expected_sent
