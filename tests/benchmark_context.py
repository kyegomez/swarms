"""
Measure the context each agent call receives, and how far one false
note travels, in shared-conversation structures and in ContextTree.

N agents work R rounds on one spec. Each agent's model is a stub that
records every request and returns a 300-character reply. A request is
the task plus the content of every message that reaches run(), and
tokens are estimated as ceil(chars / 4). The agents have no system
prompt, which would add the same to every call in every arm. In round
1, agent 0's reply opens with a sentence that contradicts the spec.
Reach counts the later requests that contain it.

Arms:
    sequential: SequentialWorkflow with max_loops=R. Each agent reads
        the whole shared conversation.
    round_robin: RoundRobinSwarm with max_loops=R. The same, plus a
        turn header.
    concurrent: one ConcurrentWorkflow run R times. Agents share
        nothing, but each re-sends its own earlier runs.
    tree: each prompt is tree.render() and the agent's name, and each
        reply is saved whole as a note. All agents are one flat team.
    tree_gated: the same, with a stub gate that is told the planted
        sentence. It shows what a gate that catches the note buys, not
        how often a real gate would.

Without a gate, ContextTree shows the planted note to every teammate
until newer notes push it out of the token budget. The budget caps
how far it travels; only the gate stops it. Below the budget, the
tree arm sends about as much as the shared conversation does.

GroupChat and HierarchicalSwarm are left out: they route work from
the model's tool calls, which a fixed text reply cannot make. Context
compression is off; it only acts near a 128k window. Nothing here
measures answer quality or how a real model treats the false note.

Run from the repo root:
    PYTHONPATH=. python tests/benchmark_context.py
    PYTHONPATH=. python tests/benchmark_context.py --agents 4 32
    PYTHONPATH=. python tests/benchmark_context.py --budget 800
"""

import argparse
import math
import os
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("SWARMS_TELEMETRY_ON", "False")

from loguru import logger

from swarms import (
    Agent,
    ConcurrentWorkflow,
    ContextTree,
    RoundRobinSwarm,
    SequentialWorkflow,
)

# Per-run INFO lines would bury the table.
logger.disable("swarms")

SPEC = (
    "Build the checkout service. An order takes at most one "
    "discount code; reject a second code with a clear error."
)
PLANTED = "Note: discount codes stack now, apply every code."
FILLER = "Cart totals, tax lines and refunds were checked. " * 7
REPLY_CHARS = 300
ARMS = [
    "sequential",
    "round_robin",
    "concurrent",
    "tree",
    "tree_gated",
]


class StubModel:
    """Model stand in that records each request it receives."""

    def __init__(self, name: str, log: List[str]):
        self.name = name
        self.log = log
        self.calls = 0

    def run(
        self,
        task: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        **kwargs,
    ) -> str:
        """
        Record the request and return a 300-character reply.

        Args:
            task (Optional[str]): The task, if the agent sent one.
            messages (Optional[List[dict]]): The chat turns sent.

        Returns:
            str: The reply. Agent 0's first opens with PLANTED.
        """
        parts = [task] if task else []
        for message in messages or []:
            if message.get("content"):
                parts.append(str(message["content"]))
        self.log.append("\n".join(parts))
        self.calls += 1
        lead = f"{self.name} update {self.calls}."
        if self.name == "agent-0" and self.calls == 1:
            lead = PLANTED
        return f"{lead} {FILLER}"[:REPLY_CHARS]


def make_agents(n: int, log: List[str]) -> List[Agent]:
    """
    Build n agents whose stub models share one request log.

    Args:
        n (int): Number of agents.
        log (List[str]): Receives every request, in call order.

    Returns:
        List[Agent]: Agents named agent-0 to agent-{n-1}.
    """
    return [
        Agent(
            agent_name=f"agent-{i}",
            llm=StubModel(f"agent-{i}", log),
            output_type="final",
            print_on=False,
            context_compression=False,
        )
        for i in range(n)
    ]


def run_tree(
    agents: List[Agent], rounds: int, budget: int, gated: bool
) -> None:
    """
    Run the agents in turn; each reads the tree and writes a note.

    Args:
        agents (List[Agent]): The team, all at one level of one task.
        rounds (int): Turns per agent.
        budget (int): Tokens of notes each render returns.
        gated (bool): Whether a stub gate blocks the planted note.
    """

    def gate(author: str, text: str) -> Tuple[bool, str]:
        # A stub that blocks the planted sentence and nothing else.
        return PLANTED not in text, "contradicts the spec"

    notes = ContextTree(
        spec=SPEC, gate=gate if gated else None, token_budget=budget
    )
    for _ in range(rounds):
        for agent in agents:
            context = notes.render()
            prompt = f"{context}\n\nYou are {agent.agent_name}."
            # messages=[] stops the agent re-sending its earlier runs.
            reply = agent.run(prompt, messages=[])
            try:
                notes.write(reply, author=agent.agent_name)
            except ValueError as error:
                # Only the gate may drop a note; anything else is a bug.
                if not str(error).startswith("Blocked by the gate"):
                    raise


def run_arm(
    arm: str, agents: List[Agent], rounds: int, budget: int
) -> None:
    """
    Run one arm for all rounds.

    Args:
        arm (str): One of ARMS.
        agents (List[Agent]): The team.
        rounds (int): Turns per agent.
        budget (int): Tokens of notes each tree render returns.
    """
    if arm == "sequential":
        SequentialWorkflow(agents=agents, max_loops=rounds).run(SPEC)
    elif arm == "round_robin":
        RoundRobinSwarm(agents=agents, max_loops=rounds).run(SPEC)
    elif arm == "concurrent":
        workflow = ConcurrentWorkflow(agents=agents)
        for _ in range(rounds):
            workflow.run(SPEC)
    else:
        run_tree(agents, rounds, budget, gated=arm == "tree_gated")


def measure(log: List[str], n: int) -> Dict[str, float]:
    """
    Summarise one arm's requests.

    Args:
        log (List[str]): Every request, in call order.
        n (int): Calls per round, so the log splits into rounds.

    Returns:
        Dict[str, float]: Calls, total, mean and max tokens, growth
            from the first round's largest call to the last's, reach
            of the planted sentence and share of calls with the spec.
    """
    tokens = [math.ceil(len(text) / 4) for text in log]
    return {
        "calls": len(log),
        "total": sum(tokens),
        "mean": sum(tokens) / len(tokens),
        "max": max(tokens),
        "growth": max(tokens[-n:]) / max(tokens[:n]),
        "reach": sum(PLANTED in text for text in log),
        "spec": sum(SPEC in text for text in log) / len(log),
    }


def main() -> None:
    """Run the sweep and print one row per size and arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agents", type=int, nargs="+", default=[4, 8, 16]
    )
    parser.add_argument(
        "--rounds", type=int, nargs="+", default=[3, 6]
    )
    parser.add_argument("--budget", type=int, default=1500)
    args = parser.parse_args()

    print(
        f"{'N':>3}{'R':>3}  {'arm':<12}{'calls':>6}{'total tok':>10}"
        f"{'mean':>7}{'max':>7}{'growth':>8}{'reach':>7}{'spec':>6}"
    )
    for n in args.agents:
        for rounds in args.rounds:
            for arm in ARMS:
                log: List[str] = []
                run_arm(arm, make_agents(n, log), rounds, args.budget)
                # Rounds are cut from the log by position.
                assert len(log) == n * rounds, f"{arm}: {len(log)}"
                m = measure(log, n)
                print(
                    f"{n:>3}{rounds:>3}  {arm:<12}{m['calls']:>6}"
                    f"{m['total']:>10}{m['mean']:>7.0f}{m['max']:>7}"
                    f"{m['growth']:>8.1f}{m['reach']:>7}"
                    f"{m['spec']:>6.0%}"
                )


if __name__ == "__main__":
    main()
