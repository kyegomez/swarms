r"""
Topology 4 — Multi-provider ensemble (parallel models → aggregator)
==================================================================

Three workers run the same task in parallel, each backed by a different
LLM (mixing OpenAI and Anthropic). An aggregator merges their answers
into a single high-confidence response.

    Haiku_Worker, GPT4o_Worker, Sonnet_Worker -> Aggregator
    \____________________parallel__________________/

Requires OPENAI_API_KEY and ANTHROPIC_API_KEY in the env.
"""

import time
from swarms import Agent, AgentRearrange


def _agent(name: str, model: str, prompt: str) -> Agent:
    return Agent(
        agent_name=name,
        system_prompt=prompt,
        model_name=model,
        max_loops=1,
        verbose=False,
        persistent_memory=False,
    )


common_prompt = (
    "You are an expert. Answer the user's question in 3 concise sentences. "
    "Be specific and avoid hedging."
)

haiku = _agent("Haiku_Worker", "claude-haiku-4-5-20251001", common_prompt)
gpt4o = _agent("GPT4o_Worker", "gpt-4o-mini", common_prompt)
sonnet = _agent("Sonnet_Worker", "claude-sonnet-4-6", common_prompt)
aggregator = _agent(
    "Aggregator",
    "gpt-4.1",
    "You are a senior editor. Given three independent expert answers, "
    "produce one synthesized answer that captures the strongest claims "
    "from each source. Cite which sources agreed on the key claim.",
)

pipeline = AgentRearrange(
    name="multi-provider-ensemble",
    agents=[haiku, gpt4o, sonnet, aggregator],
    flow="Haiku_Worker, GPT4o_Worker, Sonnet_Worker -> Aggregator",
    max_loops=1,
    output_type="dict",
    autosave=False,
)

TASK = (
    "What is the single most important architectural decision when designing "
    "a low-latency online inference service for a 70B-parameter LLM?"
)


def main() -> None:
    print("=" * 72)
    print(f"MULTI-PROVIDER ENSEMBLE  |  flow: {pipeline.flow}")
    print("=" * 72)
    print(f"Task: {TASK}\n")

    t0 = time.perf_counter()
    messages = pipeline.run(TASK)
    print(f"Completed in {time.perf_counter() - t0:.2f}s\n")

    order = ["Haiku_Worker", "GPT4o_Worker", "Sonnet_Worker", "Aggregator"]
    latest = {}
    for msg in messages:
        role = msg.get("role")
        if role in order:
            latest[role] = msg.get("content", "")

    for name in order:
        out = latest.get(name)
        if not out:
            continue
        print("-" * 72)
        print(f"[{name}]")
        print("-" * 72)
        print(str(out).strip())
        print()


if __name__ == "__main__":
    main()
