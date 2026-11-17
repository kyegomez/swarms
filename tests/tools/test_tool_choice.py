import litellm
import pytest

from swarms import (
    Agent,
    HeavySwarm,
    HierarchicalSwarm,
    PlannerWorkerSwarm,
)
from swarms.structs.groupchat import RESPOND_TOOL, GroupChat
from swarms.tools.tool_choice import forced_tool_choice

SCHEMA = {"type": "function", "function": {"name": "respond"}}
FORCED = {"type": "function", "function": {"name": "respond"}}
NO_FORCE = "anthropic/test-model-without-forced-tools"


@pytest.fixture(autouse=True)
def listed_models(monkeypatch):
    """Two models in LiteLLM's model list, one that rejects a forced choice."""
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
        _bid_request,
    ],
    ids=["planner", "director", "questions", "groupchat_bid"],
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
