"""
Real-time token streaming from a single Agent.

Two ways to consume the stream — both work for any max_loops value
(1, > 1 with tools, or "auto" autonomous mode):

    1. agent.run_stream(task)   -> sync generator yielding str tokens
    2. agent.arun_stream(task)  -> async generator yielding str tokens

Tokens flow through every internal loop: tool-call turns, synthesis turns
after a tool returns, and the plan/execute/summary phases when
max_loops="auto".

Run:
    export OPENAI_API_KEY=sk-...
    python examples/streaming/agent_streaming.py
"""

import asyncio
import sys

from swarms import Agent


def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b


# ---------------------------------------------------------------------------
# 1. Sync streaming with a multi-loop agent that calls a tool.
#    Tokens stream during the tool-call turn AND the synthesis turn.
# ---------------------------------------------------------------------------
def sync_streaming_with_tool() -> None:
    print("=== sync run_stream — max_loops=3 with a tool ===\n")
    agent = Agent(
        agent_name="Calculator",
        model_name="gpt-5.4-mini",
        max_loops=3,
        tools=[add],
        persistent_memory=False,
        print_on=False,
    )
    for token in agent.run_stream(
        "Use the add tool to compute 17 + 25, then state the result."
    ):
        sys.stdout.write(token)
        sys.stdout.flush()
    print("\n")


# ---------------------------------------------------------------------------
# 2. Async streaming with a single-loop agent.
#    Drop-in for any async caller.
# ---------------------------------------------------------------------------
async def async_streaming() -> None:
    print("=== async arun_stream — max_loops=1 ===\n")
    agent = Agent(
        agent_name="Writer",
        model_name="gpt-5.4-mini",
        max_loops=1,
        persistent_memory=False,
        print_on=False,
    )
    async for token in agent.arun_stream(
        "Explain the difference between concurrency and parallelism in two sentences."
    ):
        sys.stdout.write(token)
        sys.stdout.flush()
    print("\n")


# ---------------------------------------------------------------------------
# 3. Async streaming through the autonomous plan→execute→summary loop.
#    Tokens stream for every phase, including the final summary.
# ---------------------------------------------------------------------------
async def async_streaming_autonomous() -> None:
    print(
        "=== async arun_stream — max_loops='auto' with a tool ===\n"
    )
    agent = Agent(
        agent_name="AutoBot",
        model_name="gpt-5.4-mini",
        max_loops="auto",
        tools=[add],
        persistent_memory=False,
        print_on=False,
    )
    async for token in agent.arun_stream(
        "Use the add tool to compute 99 + 1, then briefly explain the answer."
    ):
        sys.stdout.write(token)
        sys.stdout.flush()
    print("\n")


async def main() -> None:
    sync_streaming_with_tool()
    await async_streaming()
    await async_streaming_autonomous()


if __name__ == "__main__":
    asyncio.run(main())
