"""
AgentRearrange diamond-flow demo with REAL LLM agents
=====================================================

This example wires four real `Agent` instances into a diamond:

    Planner -> Coder, Reviewer -> Tester

- Planner runs first.
- Coder and Reviewer run concurrently on the planner's output.
- Tester runs last, with access to everything that came before.

Requires ``OPENAI_API_KEY`` in the environment. Run with:

    python examples/multi_agent/agent_rearrange_examples/diamond_flow_real_agents.py
"""

import time

from swarms import Agent, AgentRearrange


MODEL = "gpt-4o-mini"

planner = Agent(
    agent_name="Planner",
    system_prompt=(
        "You are a senior engineer. Given a coding task, produce a tight, "
        "numbered plan in 5 bullet points or fewer. No code yet."
    ),
    model_name=MODEL,
    max_loops=1,
    verbose=False,
    persistent_memory=False,
)

coder = Agent(
    agent_name="Coder",
    system_prompt=(
        "You are an implementation expert. Given a plan, write the Python "
        "code that satisfies it. Output the function only, in a fenced "
        "code block, no commentary."
    ),
    model_name=MODEL,
    max_loops=1,
    verbose=False,
    persistent_memory=False,
)

reviewer = Agent(
    agent_name="Reviewer",
    system_prompt=(
        "You are a code reviewer. Given a plan, list potential bugs, "
        "edge cases, and ambiguities the implementer should watch for. "
        "Three concise bullets."
    ),
    model_name=MODEL,
    max_loops=1,
    verbose=False,
    persistent_memory=False,
)

tester = Agent(
    agent_name="Tester",
    system_prompt=(
        "You are a test author. Given the plan, the implementation, and "
        "the reviewer's notes, write three pytest test cases (happy path, "
        "edge case, failure case). Code only, in a fenced block."
    ),
    model_name=MODEL,
    max_loops=1,
    verbose=False,
    persistent_memory=False,
)

pipeline = AgentRearrange(
    name="diamond-demo",
    agents=[planner, coder, reviewer, tester],
    flow="Planner -> Coder, Reviewer -> Tester",
    max_loops=1,
    team_awareness=True,
    output_type="dict",  # list of {"role": agent_name, "content": ...} messages
    autosave=False,
)

TASK = (
    "Write a Python function `validate_email(address: str) -> bool` that "
    "returns True only for well-formed RFC-5322-style email addresses. "
    "Reject empty strings, missing @, missing domain, or whitespace."
)


def main() -> None:
    print("=" * 72)
    print("AgentRearrange diamond demo — real LLM agents")
    print(f"  flow:  {pipeline.flow}")
    print(f"  model: {MODEL}")
    print("=" * 72)
    print()
    print(f"Task: {TASK}\n")

    t0 = time.perf_counter()
    messages = pipeline.run(TASK)
    elapsed = time.perf_counter() - t0

    print(f"\nCompleted in {elapsed:.2f}s\n")

    # Collect the last message produced by each agent (in flow order).
    agent_outputs = {}
    for msg in messages:
        role = msg.get("role")
        if role in {"Planner", "Coder", "Reviewer", "Tester"}:
            agent_outputs[role] = msg.get("content", "")

    print("=" * 72)
    print("Per-agent outputs")
    print("=" * 72)
    for agent_name in ["Planner", "Coder", "Reviewer", "Tester"]:
        output = agent_outputs.get(agent_name)
        if not output:
            continue
        print("-" * 72)
        print(f"[{agent_name}]")
        print("-" * 72)
        print(str(output).strip())
        print()


if __name__ == "__main__":
    main()
