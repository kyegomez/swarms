"""
agent.usage — what a run cost, from the provider's own numbers.

Every non-streaming completion carries the provider's exact token counts.
The agent sums them over every call it makes, including the per-loop tool
summary calls, and exposes the total as ``agent.usage``:

    {"input_tokens": ..., "output_tokens": ..., "cached_tokens": ..., "total_tokens": ...}

``cached_tokens`` is the part of ``input_tokens`` the provider served from
its prompt cache — already included in ``input_tokens``, not extra.

Totals are lifetime totals for the agent: they keep growing across runs.
Snapshot before and after if you want the cost of one run.

Run:
    export OPENAI_API_KEY=...
    python examples/single_agent/utils/agent_usage.py
"""

from swarms import Agent

# Published rates in $ per 1M tokens. Check your provider's price list.
INPUT_PRICE = 2.50
CACHED_INPUT_PRICE = 1.25
OUTPUT_PRICE = 10.00


def cost(usage: dict) -> float:
    """Dollar cost of a usage dict at the rates above."""
    uncached = usage["input_tokens"] - usage["cached_tokens"]
    return (
        uncached * INPUT_PRICE
        + usage["cached_tokens"] * CACHED_INPUT_PRICE
        + usage["output_tokens"] * OUTPUT_PRICE
    ) / 1_000_000


def show(label: str, usage: dict) -> None:
    print(
        f"{label:<28} in={usage['input_tokens']:>6}  "
        f"cached={usage['cached_tokens']:>6}  "
        f"out={usage['output_tokens']:>6}  "
        f"total={usage['total_tokens']:>6}  ${cost(usage):.5f}"
    )


# --- 1. One call ------------------------------------------------------------

agent = Agent(
    agent_name="Usage-Demo",
    system_prompt="You answer in one short paragraph.",
    model_name="gpt-5.4",
    max_loops=1,
)

show("before any run", agent.usage)

agent.run(
    "Explain what a token is, for someone new to language models."
)
show("after one run", agent.usage)


# --- 2. Several loops with a tool: every call counts ------------------------


def word_count(text: str) -> int:
    """Count the words in a piece of text.

    Args:
        text: The text to count.
    """
    return len(text.split())


tool_agent = Agent(
    agent_name="Usage-Tool-Demo",
    system_prompt="Use the word_count tool when asked about length.",
    model_name="gpt-5.4",
    tools=[word_count],
    max_loops=3,
)

tool_agent.run(
    "How many words are in this sentence: 'The quick brown fox jumps "
    "over the lazy dog'? Then tell me whether that is a long sentence."
)
# Each loop that calls a tool makes two provider calls: the tool call
# and the summary of its result. Both land here.
show("tool agent, 3 loops", tool_agent.usage)


# --- 3. The cost of one run, on an agent that has run before ----------------

before = agent.usage
agent.run("Now explain a context window, the same way.")
after = agent.usage

per_run = {key: after[key] - before[key] for key in after}
show("second run only", per_run)
show("lifetime", after)
