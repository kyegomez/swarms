r"""
Topology 1 — Fan-out (one → many)
=================================

A single source agent produces a piece of content, then three downstream
agents fan out in parallel to localize it. No final join.

    Source -> Translator_FR, Translator_ES, Translator_JP
              \_____________parallel step_____________/

Each translator sees the source paragraph and produces an independent
localized version. Results are returned in dict form, one entry per agent.
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


source = _agent(
    "Source",
    "Write ONE concise English paragraph (≤60 words) introducing the "
    "concept of multi-agent orchestration to a non-technical reader.",
)
fr = _agent(
    "Translator_FR",
    "Translate the most recent English paragraph into formal French. "
    "Output only the translation.",
)
es = _agent(
    "Translator_ES",
    "Translate the most recent English paragraph into Latin-American Spanish. "
    "Output only the translation.",
)
jp = _agent(
    "Translator_JP",
    "Translate the most recent English paragraph into natural Japanese (keigo). "
    "Output only the translation.",
)

pipeline = AgentRearrange(
    name="fan-out",
    agents=[source, fr, es, jp],
    flow="Source -> Translator_FR, Translator_ES, Translator_JP",
    max_loops=1,
    output_type="dict",
    autosave=False,
)


def main() -> None:
    print("=" * 72)
    print(f"FAN-OUT  |  flow: {pipeline.flow}")
    print("=" * 72)

    t0 = time.perf_counter()
    messages = pipeline.run("Introduce multi-agent orchestration in one paragraph.")
    print(f"\nCompleted in {time.perf_counter() - t0:.2f}s\n")

    latest = {}
    for msg in messages:
        role = msg.get("role")
        if role in {"Source", "Translator_FR", "Translator_ES", "Translator_JP"}:
            latest[role] = msg.get("content", "")

    for name in ["Source", "Translator_FR", "Translator_ES", "Translator_JP"]:
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
