import json
import threading

from swarms.structs.agent import Agent
from swarms.structs.hybrid_hiearchical_peer_swarm import (
    HybridHierarchicalClusterSwarm,
)


class Router:
    def reset(self):
        pass

    def run(self, task):
        return json.dumps(
            {
                "reasoning": "There is one swarm.",
                "swarm_name": "Writers",
                "task_description": task,
            }
        )


class Writers:
    name = "Writers"
    description = "Answers the task"
    agents = ()
    swarm_type = "SequentialWorkflow"
    barrier = None

    def run(self, task):
        if self.barrier:
            self.barrier.wait()
        return f"answer to {task}"


def contents(result):
    return [message["content"] for message in result]


def select_writers(task_description):
    return {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "select_swarm",
            "arguments": json.dumps(
                {
                    "reasoning": "There is one swarm.",
                    "swarm_name": "Writers",
                    "task_description": task_description,
                }
            ),
        },
    }


def test_each_task_gets_its_own_conversation():
    writers = Writers()
    swarm = HybridHierarchicalClusterSwarm(swarms=[writers])
    swarm.router_agent = Router()

    swarm.run("first")

    assert contents(swarm.run("second")) == [
        "second",
        "answer to second",
    ]

    writers.barrier = threading.Barrier(2, timeout=10)

    assert [contents(r) for r in swarm.batched_run(["a", "b"])] == [
        ["a", "answer to a"],
        ["b", "answer to b"],
    ]


def test_run_routes_a_select_swarm_tool_call(monkeypatch):
    swarm = HybridHierarchicalClusterSwarm(swarms=[Writers()])
    swarm.router_agent.print_on = False
    monkeypatch.setattr(
        swarm.router_agent,
        "call_llm",
        lambda *args, **kwargs: [select_writers("write it")],
    )

    assert contents(swarm.run("task")) == [
        "task",
        "answer to write it",
    ]


def test_router_sees_only_the_current_task(monkeypatch):
    barrier = threading.Barrier(1)

    def route_what_the_router_was_asked(agent, *args, **kwargs):
        barrier.wait()
        asked = [
            message["content"]
            for message in agent.short_memory.conversation_history
            if message["role"] == agent.user_name
        ]
        return [select_writers(" ".join(asked))]

    monkeypatch.setattr(
        Agent, "call_llm", route_what_the_router_was_asked
    )
    swarm = HybridHierarchicalClusterSwarm(swarms=[Writers()])
    swarm.router_agent.print_on = False

    swarm.run("first")

    assert contents(swarm.run("second")) == [
        "second",
        "answer to second",
    ]

    barrier = threading.Barrier(2, timeout=10)

    assert [contents(r) for r in swarm.batched_run(["a", "b"])] == [
        ["a", "answer to a"],
        ["b", "answer to b"],
    ]
