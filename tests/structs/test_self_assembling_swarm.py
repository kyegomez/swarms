import itertools
import json
import random

import litellm
import pytest

from swarms import Agent, SelfAssemblingSwarm


def _fake_value(schema, counter, rng):
    """Build a value that fits a JSON schema, for a scripted tool call."""
    kind = schema.get("type")

    if "enum" in schema:
        return rng.choice(schema["enum"])
    if kind == "string":
        return f"point {next(counter)} about the task"
    if kind == "number":
        return round(
            rng.uniform(
                schema.get("minimum", 0), schema.get("maximum", 1)
            ),
            2,
        )
    if kind == "integer":
        return int(schema.get("minimum", 1))
    if kind == "boolean":
        return True
    if kind == "array":
        item = schema.get("items", {"type": "string"})
        count = max(schema.get("minItems", 2), 1)
        return [_fake_value(item, counter, rng) for _ in range(count)]
    if kind == "object":
        return {
            key: _fake_value(value, counter, rng)
            for key, value in schema.get("properties", {}).items()
        }
    return "x"


@pytest.fixture
def scripted_model(monkeypatch, tmp_path):
    """Answer every call with a tool call that fits the offered schema."""
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))
    counter = itertools.count()
    rng = random.Random(0)

    def completion(**params):
        tools = params.get("tools") or []
        if not tools:
            return litellm.completion(
                **params, mock_response=f"answer {next(counter)}"
            )

        choice = params.get("tool_choice")
        if isinstance(choice, dict):
            wanted = choice["function"]["name"]
            tool = next(
                t for t in tools if t["function"]["name"] == wanted
            )
        else:
            tool = rng.choice(tools)

        schema = tool["function"].get(
            "parameters", {"type": "object"}
        )
        arguments = _fake_value(schema, counter, rng)
        return litellm.completion(
            **params,
            mock_tool_calls=[
                {
                    "id": f"call-{next(counter)}",
                    "type": "function",
                    "function": {
                        "name": tool["function"]["name"],
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
        )

    monkeypatch.setattr(
        "swarms.utils.litellm_wrapper.completion", completion
    )


@pytest.mark.parametrize(
    "use_verifier", [True, False], ids=["verifier", "llm judge"]
)
def test_search_returns_an_answer_and_a_reusable_harness(
    scripted_model, use_verifier
):
    names = ["Researcher", "Analyst", "Writer", "Critic"]
    agents = [
        Agent(
            agent_name=name,
            system_prompt=f"You are the {name}.",
            model_name="gpt-4o-mini",
            max_loops=1,
            print_on=False,
        )
        for name in names
    ]
    scored = []

    def verifier(task, answer):
        scored.append(answer)
        return 0.7

    swarm = SelfAssemblingSwarm(
        agents=agents,
        max_loops=2,
        seed=0,
        judge_model_name="gpt-4o-mini",
        verifier=verifier if use_verifier else None,
    )

    answer = swarm.run(
        "Compare PostgreSQL and MongoDB for a SaaS app."
    )

    harness = swarm.harness
    subgoals = {subgoal["id"] for subgoal in harness["subgoals"]}
    assert isinstance(answer, str) and answer
    assert {team["subgoal"] for team in harness["teams"]} == subgoals
    assert {
        member["name"]
        for team in harness["teams"]
        for member in team["members"]
    } <= set(names)
    assert bool(scored) is use_verifier

    scored.clear()
    rerun = swarm.run_harness(
        "Compare MySQL and DynamoDB for a SaaS app."
    )

    assert isinstance(rerun, str) and rerun
    assert scored == []
