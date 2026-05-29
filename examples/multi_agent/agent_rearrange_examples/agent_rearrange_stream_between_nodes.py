"""
AgentRearrange: stream_between_nodes example.

Demonstrates the opt-in pipeline streaming feature where downstream
agents begin generating from partial upstream output rather than waiting
for the upstream agent to finish.

A 4-agent research pipeline is run three times — once per buffer strategy
— so you can compare the token interleaving behaviour:

  Researcher -> Analyst -> Writer -> Editor

  buffer_strategy="line"   — downstream starts after first newline from upstream
  buffer_strategy="tokens" — downstream starts after every 50 tokens from upstream
  buffer_strategy="all"    — no pipeline benefit; same as standard streaming

Run:
    python examples/multi_agent/agent_rearrange_examples/agent_rearrange_stream_between_nodes.py
"""

import sys
import time

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from swarms import Agent
from swarms.structs.agent_rearrange import AgentRearrange

TASK = (
    "Analyse the impact of large language models on software engineering "
    "productivity. Cover: (1) code generation, (2) debugging assistance, "
    "(3) documentation."
)


def make_pipeline(
    buffer_strategy: str,
    buffer_token_count: int = 50,
) -> AgentRearrange:
    researcher = Agent(
        agent_name="Researcher",
        agent_description="Research and gather key facts on the topic.",
        model_name="claude-sonnet-4-5",
        max_loops=1,
        verbose=False,
        print_on=False,
    )
    analyst = Agent(
        agent_name="Analyst",
        agent_description="Analyse the facts and identify key trends.",
        model_name="claude-sonnet-4-5",
        max_loops=1,
        verbose=False,
        print_on=False,
    )
    writer = Agent(
        agent_name="Writer",
        agent_description="Write a concise report from the analysis.",
        model_name="claude-sonnet-4-5",
        max_loops=1,
        verbose=False,
        print_on=False,
    )
    editor = Agent(
        agent_name="Editor",
        agent_description="Edit the report for clarity and conciseness.",
        model_name="claude-sonnet-4-5",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    return AgentRearrange(
        agents=[researcher, analyst, writer, editor],
        flow="Researcher -> Analyst -> Writer -> Editor",
        stream_between_nodes=True,
        buffer_strategy=buffer_strategy,
        buffer_token_count=buffer_token_count,
    )


def run_demo(strategy: str, token_count: int = 50) -> None:
    print(f"\n{'=' * 60}")
    print(f"  buffer_strategy = '{strategy}'")
    if strategy == "tokens":
        print(f"  buffer_token_count = {token_count}")
    print("=" * 60)

    pipeline = make_pipeline(strategy, token_count)
    start = time.monotonic()

    current_agent = None
    total_tokens = 0

    for agent_name, token in pipeline.run_stream(
        task=TASK, with_events=False
    ):
        if agent_name != current_agent:
            if current_agent is not None:
                print()  # newline between agents
            print(f"\n[{agent_name}] ", end="", flush=True)
            current_agent = agent_name
        print(token, end="", flush=True)
        total_tokens += 1

    elapsed = time.monotonic() - start
    print(
        f"\n\nTotal tokens: {total_tokens} | Wall time: {elapsed:.2f}s"
    )


if __name__ == "__main__":
    print("AgentRearrange — stream_between_nodes demo")
    print("Flow: Researcher -> Analyst -> Writer -> Editor")
    print(f"Task: {TASK[:80]}...")

    run_demo("line")
    run_demo("tokens", token_count=50)
    run_demo("all")
