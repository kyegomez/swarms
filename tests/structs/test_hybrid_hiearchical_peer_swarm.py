import json
import threading

from swarms.structs.hybrid_hiearchical_peer_swarm import (
    HybridHierarchicalClusterSwarm,
)


class Router:
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


def test_router_request_forces_the_select_swarm_tool(monkeypatch):
    sent = []

    def capture(**params):
        sent.append(params)
        raise RuntimeError("captured")

    monkeypatch.setattr(
        "swarms.utils.litellm_wrapper.completion", capture
    )
    swarm = HybridHierarchicalClusterSwarm(swarms=[Writers()])

    try:
        swarm.router_agent.llm.run(task="route")
    except RuntimeError:
        pass

    assert sent[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": "select_swarm"},
    }


def test_router_routes_each_task_without_earlier_ones():
    swarm = HybridHierarchicalClusterSwarm(swarms=[Writers()])
    router = swarm.router_agent
    router.print_on = False
    requests = []

    def call_llm(*args, **kwargs):
        requests.append(router.short_memory.get_str())
        return json.dumps(
            {
                "reasoning": "There is one swarm.",
                "swarm_name": "Writers",
                "task_description": "write it",
            }
        )

    router.call_llm = call_llm

    swarm.run("first-task")
    swarm.run("second-task")

    assert "second-task" in requests[1]
    assert "first-task" not in requests[1]
