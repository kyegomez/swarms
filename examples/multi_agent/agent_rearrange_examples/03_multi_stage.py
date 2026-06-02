r"""
Topology 3 — Multi-stage: ingest → fan-out → fan-in → polish
============================================================

Five-agent compound flow combining all the basic patterns:

    Ingestor -> Tech, Business, Legal -> Synthesizer -> Editor
                \____parallel______/

- Ingestor reframes the task into an executable brief.
- Three specialists run in parallel from the brief.
- Synthesizer merges their findings.
- Editor polishes the final report.
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


ingestor = _agent(
    "Ingestor",
    "You are a research planner. Restate the user's question as a tight, "
    "actionable brief in 3 numbered bullets.",
)
tech = _agent(
    "Tech",
    "You are a technical analyst. Given the brief, give 2 concise points "
    "on the technical/engineering implications.",
)
business = _agent(
    "Business",
    "You are a business strategist. Given the brief, give 2 concise points "
    "on the market/commercial implications.",
)
legal = _agent(
    "Legal",
    "You are a regulatory counsel. Given the brief, give 2 concise points "
    "on the legal/compliance implications.",
)
synth = _agent(
    "Synthesizer",
    "Merge the Tech, Business, and Legal analyses into a coherent 4-bullet "
    "summary capturing the most important cross-cutting insights.",
)
editor = _agent(
    "Editor",
    "Polish the synthesis for clarity and concision. Keep it under 120 words. "
    "Output the final report only.",
)

pipeline = AgentRearrange(
    name="multi-stage",
    agents=[ingestor, tech, business, legal, synth, editor],
    flow="Ingestor -> Tech, Business, Legal -> Synthesizer -> Editor",
    max_loops=1,
    team_awareness=True,
    output_type="dict",
    autosave=False,
)

TASK = (
    "Should a mid-stage SaaS company in the EU adopt on-prem generative AI "
    "instead of relying on a third-party API provider?"
)


def main() -> None:
    print("=" * 72)
    print(f"MULTI-STAGE  |  flow: {pipeline.flow}")
    print("=" * 72)
    print(f"Task: {TASK}\n")

    t0 = time.perf_counter()
    messages = pipeline.run(TASK)
    print(f"Completed in {time.perf_counter() - t0:.2f}s\n")

    order = ["Ingestor", "Tech", "Business", "Legal", "Synthesizer", "Editor"]
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
