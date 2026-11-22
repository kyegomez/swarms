import json
import re
from collections import Counter

import litellm
import pytest

from swarms import Agent, SelfAssemblingSwarm
from swarms.structs.self_assembling_swarm import assemble_teams


@pytest.mark.parametrize(
    "bids, num_subtasks, max_team_size, expected",
    [
        (
            {"A": (0, 0.9), "B": (1, 0.8), "C": (0, 0.5)},
            2,
            None,
            [["A", "C"], ["B"]],
        ),
        (
            {"A": (0, 0.9), "B": (0, 0.8), "C": (0, 0.7)},
            2,
            2,
            [["A", "B"], ["C"]],
        ),
        (
            {"A": (0, 0.9), "B": (0, 0.8), "C": (0, 0.7)},
            3,
            None,
            [["A"], ["C"], ["B"]],
        ),
        (
            {"A": (0, 0.9), "B": (None, 0.0), "C": (5, 0.4)},
            2,
            None,
            [["A", "B"], ["C"]],
        ),
    ],
    ids=[
        "each agent gets its choice",
        "full subtask sends the rest to the smallest team",
        "unchosen subtask takes the largest team's lowest bidder",
        "invalid bid joins the smallest team",
    ],
)
def test_assemble_teams(bids, num_subtasks, max_team_size, expected):
    assert (
        assemble_teams(bids, num_subtasks, max_team_size) == expected
    )


def test_teams_discuss_share_results_and_are_combined(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))
    bids = {"A": (1, 0.9), "B": (2, 0.8), "C": (1, 0.4)}
    calls = Counter()
    prompts = {}

    def completion(**params):
        system = str(params["messages"][0]["content"])
        user = str(params["messages"][-1]["content"])
        tool = (params.get("tool_choice") or {}).get("function", {})
        found = re.search(r"You are agent (\w)", system)
        name = found.group(1) if found else "Coordinator"

        if tool.get("name") == "post_subtasks":
            arguments = {"subtasks": ["research", "write"]}
        elif tool.get("name") == "bid":
            subtask, score = bids[name]
            arguments = {
                "subtask": subtask,
                "score": score,
                "reason": "",
            }
        else:
            calls[name] += 1
            prompts[f"{name}#{calls[name]}"] = user
            return litellm.completion(
                **params, mock_response=f"<{name} #{calls[name]}>"
            )

        return litellm.completion(
            **params,
            mock_tool_calls=[
                {
                    "id": "call",
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
        )

    monkeypatch.setattr(
        "swarms.utils.litellm_wrapper.completion", completion
    )
    agents = [
        Agent(
            agent_name=name,
            system_prompt=f"You are agent {name}.",
            model_name="gpt-4o-mini",
            max_loops=1,
            print_on=False,
        )
        for name in "ABC"
    ]
    swarm = SelfAssemblingSwarm(
        agents=agents,
        max_loops=2,
        coordinator_model_name="gpt-4o-mini",
    )

    result = swarm.run("Write a report.")

    assert [
        (team.subtask, [m.agent_name for m in team.members])
        for team in swarm.teams
    ] == [("research", ["A", "C"]), ("write", ["B"])]
    assert "<C #1>" in prompts["A#1"]
    assert "<A #1>" in prompts["B#2"]
    assert "<B #1>" in prompts["C#2"]
    assert "<A #2>" in prompts["Coordinator#1"]
    assert "<B #2>" in prompts["Coordinator#1"]
    assert result == "<Coordinator #1>"


@pytest.mark.parametrize(
    "names, settings",
    [
        ([], {}),
        (["A", "A"], {}),
        (["A"], {"max_loops": 0}),
        (["A"], {"max_team_size": 0}),
    ],
    ids=["no agents", "shared name", "no loops", "empty teams"],
)
def test_invalid_setups_are_rejected(names, settings):
    agents = [
        Agent(
            agent_name=name, model_name="gpt-4o-mini", print_on=False
        )
        for name in names
    ]

    with pytest.raises(ValueError):
        SelfAssemblingSwarm(agents=agents, **settings)
