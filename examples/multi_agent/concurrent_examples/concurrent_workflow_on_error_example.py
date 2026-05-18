"""
ConcurrentWorkflow — on_error example
======================================

Demonstrates the two error-handling modes introduced for ConcurrentWorkflow:

  on_error="store"  (default)
      A failing agent's error is captured as a clearly-labelled entry in the
      conversation history under the role "<agent_name> (failed)".  Sibling
      agents that already completed successfully are not discarded.

  on_error="raise"
      The first agent exception propagates out of wf.run() immediately after
      all futures settle, preserving the previous fail-fast behaviour.

Requires ANTHROPIC_API_KEY to be set in the environment.
"""

import pprint

from swarms import Agent, ConcurrentWorkflow

# ---------------------------------------------------------------------------
# Scenario 1 — on_error="store"  (default, recommended)
# Two real agents complete successfully; a third is sabotaged to fail so we
# can confirm its error is stored without killing the sibling results.
# ---------------------------------------------------------------------------

print("=" * 60)
print("Scenario 1: on_error='store'  (default)")
print("=" * 60)

researcher = Agent(
    agent_name="Researcher",
    system_prompt="You are a concise research analyst. Answer in 2-3 sentences.",
    model_name="claude-sonnet-4-5",
    max_loops=1,
    print_on=False,
)

strategist = Agent(
    agent_name="Strategist",
    system_prompt="You are a strategic advisor. Answer in 2-3 sentences.",
    model_name="claude-sonnet-4-5",
    max_loops=1,
    print_on=False,
)

broken = Agent(
    agent_name="DataFetcher",
    system_prompt="You fetch data.",
    model_name="claude-sonnet-4-5",
    max_loops=1,
    print_on=False,
)
# Simulate a hard failure (e.g. a broken API, bad credentials, etc.)
broken.run = lambda *a, **kw: (_ for _ in ()).throw(
    RuntimeError("DataFetcher: upstream API unavailable")
)

wf = ConcurrentWorkflow(
    name="ResearchTeam",
    agents=[researcher, strategist, broken],
    on_error="store",   # default — can be omitted
    output_type="list",
    autosave=False,
)

result = wf.run("Summarise the current state of renewable energy adoption.")

print("\nFormatted output:")
pprint.pprint(result)

print("\nRaw conversation history:")
for entry in wf.conversation.conversation_history:
    role    = entry.get("role", "")
    content = entry.get("content", "")
    marker  = "  *** ERROR ***" if "(failed)" in role else ""
    print(f"  {role}: {content[:120]!r}{marker}")

# ---------------------------------------------------------------------------
# Scenario 2 — on_error="raise"  (opt-in, preserves old fail-fast behaviour)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Scenario 2: on_error='raise'")
print("=" * 60)

analyst = Agent(
    agent_name="Analyst",
    system_prompt="You are a market analyst. Answer in 2-3 sentences.",
    model_name="claude-sonnet-4-5",
    max_loops=1,
    print_on=False,
)

scraper = Agent(
    agent_name="Scraper",
    system_prompt="You scrape data.",
    model_name="claude-sonnet-4-5",
    max_loops=1,
    print_on=False,
)
scraper.run = lambda *a, **kw: (_ for _ in ()).throw(
    RuntimeError("Scraper: connection timed out")
)

wf2 = ConcurrentWorkflow(
    name="AnalysisTeam",
    agents=[analyst, scraper],
    on_error="raise",
    autosave=False,
)

try:
    wf2.run("What are the biggest trends in renewable energy investment?")
except RuntimeError as exc:
    print(f"\nCaught expected RuntimeError: {exc}")
    print("(on_error='raise' lets the exception escape — old behaviour preserved.)")
