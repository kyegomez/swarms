r"""
Topology 2 — Fan-in / many → one (summarize)
============================================

Three independent analysts run in parallel on the same question, then a
synthesizer agent merges their views into one recommendation.

    Bull, Bear, Quant -> Synthesizer
    \____parallel____/
                       \___ summarize ___/
"""

import time
from swarms import Agent, AgentRearrange

MODEL = "gpt-4o-mini"


def _agent(name: str, prompt: str) -> Agent:
    return Agent(
        agent_name=name,
        system_prompt=prompt,
        model_name=MODEL,
        max_loops=1,
        verbose=False,
        persistent_memory=False,
    )


bull = _agent(
    "Bull",
    "You are a bullish equity analyst. Give two concise reasons the asset "
    "will outperform over the next 12 months. No hedging.",
)
bear = _agent(
    "Bear",
    "You are a bearish equity analyst. Give two concise reasons the asset "
    "will underperform over the next 12 months. No hedging.",
)
quant = _agent(
    "Quant",
    "You are a quantitative analyst. Give two concise data-driven signals "
    "(valuation multiples, momentum, etc.) about the asset.",
)
synth = _agent(
    "Synthesizer",
    "You are a CIO. Given the prior bull, bear, and quant analyses, produce "
    "(a) a single sentence recommendation (Buy / Hold / Sell) and (b) one "
    "sentence explaining the key driver.",
)

pipeline = AgentRearrange(
    name="fan-in",
    agents=[bull, bear, quant, synth],
    flow="Bull, Bear, Quant -> Synthesizer",
    max_loops=1,
    output_type="dict",
    autosave=False,
)

TASK = "Evaluate NVIDIA (NVDA) stock for a 12-month horizon."


def main() -> None:
    print("=" * 72)
    print(f"FAN-IN  |  flow: {pipeline.flow}")
    print("=" * 72)
    print(f"Task: {TASK}\n")

    t0 = time.perf_counter()
    messages = pipeline.run(TASK)
    print(f"Completed in {time.perf_counter() - t0:.2f}s\n")

    latest = {}
    for msg in messages:
        role = msg.get("role")
        if role in {"Bull", "Bear", "Quant", "Synthesizer"}:
            latest[role] = msg.get("content", "")

    for name in ["Bull", "Bear", "Quant", "Synthesizer"]:
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
