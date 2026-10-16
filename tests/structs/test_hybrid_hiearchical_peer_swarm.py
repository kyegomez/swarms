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
