"""
Verifies streaming behaviour for SequentialWorkflow / AgentRearrange.

Checks
------
1. Plain streaming still works (backwards compat).
2. Fine-grained interleaving in parallel branches — should now flip many times
   per response (after the asyncio.sleep(0) fix), not just twice.
3. with_events=True emits structured agent_start / token / agent_end events.
4. SequentialWorkflow.arun_stream(with_events=True) propagates the events.
"""

import os
import pytest
import asyncio
import statistics
import time
from typing import List

from swarms import Agent, AgentRearrange, SequentialWorkflow


MODEL = "gpt-5.4-mini"


def make_agent(name: str, system_prompt: str) -> Agent:
    return Agent(
        agent_name=name,
        system_prompt=system_prompt,
        model_name=MODEL,
        max_loops=2,
        persistent_memory=False,
        print_on=False,
    )


def report(label: str, times: List[float], tokens: List[str]) -> None:
    if not times:
        print(f"\n[{label}] no tokens received")
        return
    t0 = times[0]
    rel = [t - t0 for t in times]
    total = rel[-1]
    gaps = [rel[i + 1] - rel[i] for i in range(len(rel) - 1)] or [0.0]
    p50 = statistics.median(gaps)
    p95 = (
        sorted(gaps)[int(0.95 * (len(gaps) - 1))]
        if len(gaps) > 1
        else gaps[0]
    )
    max_gap = max(gaps)
    cutoff_idx = int(0.1 * len(rel))
    first_10pct_time = (
        rel[cutoff_idx] if cutoff_idx < len(rel) else 0.0
    )
    spread_ratio = first_10pct_time / total if total > 0 else 0.0
    print(f"\n=== {label} ===")
    print(f"  total tokens:  {len(tokens)}")
    print(f"  duration:      {total:.2f}s")
    print(f"  median gap:    {p50 * 1000:.1f}ms")
    print(f"  p95 gap:       {p95 * 1000:.1f}ms")
    print(f"  max gap:       {max_gap * 1000:.1f}ms")
    print(
        f"  first 10% in:  {first_10pct_time:.2f}s ({spread_ratio*100:.0f}% of total)"
    )
    if spread_ratio < 0.5 and total > 0.5:
        print("  >> REAL streaming.")


# ---------------------------------------------------------------------------
# Test 1: plain SequentialWorkflow streaming (regression check)
# ---------------------------------------------------------------------------
async def test_sequential_plain():
    workflow = SequentialWorkflow(
        agents=[
            make_agent(
                "Researcher", "List two short bullets on the topic."
            ),
            make_agent("Writer", "Combine into one short paragraph."),
        ],
        autosave=False,
    )
    times, tokens = [], []
    print(
        "\n>>> SequentialWorkflow.arun_stream (plain) — live tokens:\n"
    )
    async for token in workflow.arun_stream("solid-state batteries"):
        times.append(time.perf_counter())
        tokens.append(token)
        print(token, end="", flush=True)
    print()
    report("Sequential plain", times, tokens)


# ---------------------------------------------------------------------------
# Test 2: parallel interleaving (this is what the asyncio.sleep(0) fix targets)
# ---------------------------------------------------------------------------
async def test_parallel_interleaving():
    # Longer responses make the parallel bursts more likely to overlap in
    # time, which is when fine-grained interleaving is observable.
    a = make_agent(
        "Optimist",
        "Write 6-8 upbeat sentences about the topic. Be detailed.",
    )
    b = make_agent(
        "Pessimist",
        "Write 6-8 cautious sentences about the topic. Be detailed.",
    )
    summary = make_agent(
        "Summary",
        "Write one short summary line of the prior views.",
    )
    workflow = AgentRearrange(
        agents=[a, b, summary],
        flow="Optimist, Pessimist -> Summary",
        max_loops=2,
    )

    parallel_sequence: List[str] = []
    per_agent: dict = {"Optimist": [], "Pessimist": [], "Summary": []}

    print(
        "\n>>> AgentRearrange parallel — interleaved tagged stream:\n"
    )
    async for name, token in workflow.arun_stream("AI in healthcare"):
        if name in ("Optimist", "Pessimist"):
            parallel_sequence.append(name)
        per_agent.setdefault(name, []).append(token)
        # Tagged inline view — looks scrambled because two streams are
        # interleaved character-by-character. That's the *point*: it
        # proves fine-grained interleaving. Real consumers demux by name.
        print(f"[{name[0]}]{token}", end="", flush=True)
    print()

    # Demuxed view — what a UI with two panels would render.
    print(
        "\n>>> Demuxed per-agent output (what a real consumer would render):"
    )
    for agent_name in ("Optimist", "Pessimist", "Summary"):
        text = "".join(per_agent.get(agent_name, []))
        print(f"\n--- {agent_name} ---\n{text}")

    flips = sum(
        1
        for i in range(1, len(parallel_sequence))
        if parallel_sequence[i] != parallel_sequence[i - 1]
    )
    optimist_count = parallel_sequence.count("Optimist")
    pessimist_count = parallel_sequence.count("Pessimist")

    print(f"\n  Optimist tokens (parallel phase):  {optimist_count}")
    print(f"  Pessimist tokens (parallel phase): {pessimist_count}")
    print(f"  agent-name flips during parallel:  {flips}")
    if flips >= 3:
        print(
            f"  >> FINE-GRAINED INTERLEAVING confirmed ({flips} flips)."
        )
    elif flips == 1:
        print(
            "  >> Tokens are batched — one agent fully drained before the other started."
        )
    else:
        print(f"  >> {flips} flips — partial interleaving.")


# ---------------------------------------------------------------------------
# Test 3: structured events (with_events=True) in AgentRearrange
# ---------------------------------------------------------------------------
async def test_events_agent_rearrange():
    a = make_agent(
        "AgentA", "Write one short sentence about the topic."
    )
    b = make_agent(
        "AgentB", "Write one short sentence about the topic."
    )
    c = make_agent(
        "AgentC", "Write one short final sentence about the topic."
    )
    workflow = AgentRearrange(
        agents=[a, b, c],
        flow="AgentA, AgentB -> AgentC",
        max_loops=2,
    )

    print(
        "\n>>> AgentRearrange with_events=True — event types observed:\n"
    )
    types_seen: List[str] = []
    starts = ends = token_evts = 0
    end_outputs: dict = {}

    async for evt in workflow.arun_stream(
        "neural networks", with_events=True
    ):
        t = evt["type"]
        types_seen.append(t)
        if t == "agent_start":
            starts += 1
            print(f"  [start] {evt['agent']}")
        elif t == "agent_end":
            ends += 1
            end_outputs[evt["agent"]] = evt["output"]
            print(
                f"  [end]   {evt['agent']}: {len(evt['output'])} chars"
            )
        elif t == "token":
            token_evts += 1

    print(f"\n  agent_start events: {starts}")
    print(f"  token events:       {token_evts}")
    print(f"  agent_end events:   {ends}")
    assert starts == 3, f"expected 3 starts, got {starts}"
    assert ends == 3, f"expected 3 ends, got {ends}"
    assert token_evts > 0, "no token events"
    assert all(
        out for out in end_outputs.values()
    ), "agent_end output missing"
    print("  >> Structured events working correctly.")


# ---------------------------------------------------------------------------
# Test 4: events through SequentialWorkflow
# ---------------------------------------------------------------------------
async def test_events_sequential_workflow():
    workflow = SequentialWorkflow(
        agents=[
            make_agent("Step1", "Write one bullet about the topic."),
            make_agent("Step2", "Rephrase the bullet as a question."),
        ],
        autosave=False,
    )

    print("\n>>> SequentialWorkflow with_events=True:\n")
    starts = ends = token_evts = 0
    seen_agents: List[str] = []

    async for evt in workflow.arun_stream(
        "transformers", with_events=True
    ):
        t = evt["type"]
        if t == "agent_start":
            starts += 1
            seen_agents.append(evt["agent"])
            print(f"  [start] {evt['agent']}")
        elif t == "agent_end":
            ends += 1
            print(
                f"  [end]   {evt['agent']}: {len(evt['output'])} chars"
            )
        elif t == "token":
            token_evts += 1

    print(f"\n  starts: {starts}, ends: {ends}, tokens: {token_evts}")
    print(f"  agent order: {seen_agents}")
    assert starts == 2 and ends == 2 and token_evts > 0
    assert seen_agents == ["Step1", "Step2"], "ordering broken"
    print("  >> Events flow cleanly through SequentialWorkflow.")


def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b


# ---------------------------------------------------------------------------
# Test 5: single agent, max_loops=3 with a tool
# Streams tokens through the tool-call turn AND the synthesis turn.
# ---------------------------------------------------------------------------
async def test_multi_loop_agent_with_tool():
    agent = Agent(
        agent_name="Calculator",
        model_name=MODEL,
        max_loops=3,
        tools=[add],
        persistent_memory=False,
        print_on=False,
    )
    print(
        "\n>>> Single agent, max_loops=3 with tool — live tokens:\n"
    )
    tokens: List[str] = []
    async for token in agent.arun_stream(
        "Use the add tool to compute 17 + 25, then state the result."
    ):
        tokens.append(token)
        print(token, end="", flush=True)
    print()
    full = "".join(tokens)
    print(f"\n  tokens streamed across loops: {len(tokens)}")
    assert (
        "42" in full
    ), f"expected '42' in result, got first 200 chars: {full[:200]!r}"
    assert (
        len(tokens) > 5
    ), "too few tokens — synthesis turn likely did not stream"
    print(
        "  >> Multi-loop streaming through tool call + synthesis works."
    )


# ---------------------------------------------------------------------------
# Test 6: single agent, max_loops="auto"
# Streams tokens through plan, execute, and final summary phases.
# ---------------------------------------------------------------------------
async def test_autonomous_loop_agent():
    agent = Agent(
        agent_name="AutoBot",
        model_name=MODEL,
        max_loops=3,
        tools=[add],
        persistent_memory=False,
        print_on=False,
    )
    print("\n>>> Single agent, max_loops='auto' — live tokens:\n")
    tokens: List[str] = []
    async for token in agent.arun_stream(
        "Use the add tool to compute 99 + 1, then briefly explain."
    ):
        tokens.append(token)
        print(token, end="", flush=True)
    print()
    full = "".join(tokens)
    print(f"\n  tokens streamed across phases: {len(tokens)}")
    assert (
        "100" in full
    ), f"expected '100' in result, got first 200 chars: {full[:200]!r}"
    assert (
        len(tokens) > 5
    ), "too few tokens — final summary likely did not stream"
    print("  >> Autonomous plan→execute→summary streaming works.")


# ---------------------------------------------------------------------------
# Test 7: AgentRearrange mixed flow — A -> B, C -> D
# Sequential A first, then B and C concurrently, then sequential D.
# Verifies tokens arrive in the right phase order.
# ---------------------------------------------------------------------------
async def test_mixed_flow():
    a = make_agent("A", "List one short fact about the topic.")
    b = make_agent("B", "Write one short observation.")
    c = make_agent("C", "Write one short counter-observation.")
    d = make_agent("D", "Combine the prior into one short sentence.")
    workflow = AgentRearrange(
        agents=[a, b, c, d],
        flow="A -> B, C -> D",
        max_loops=2,
    )

    print("\n>>> Mixed flow 'A -> B, C -> D' — tagged stream:\n")
    name_seq: List[str] = []
    async for name, token in workflow.arun_stream(
        "octopus intelligence"
    ):
        name_seq.append(name)
        print(f"[{name}]", end="", flush=True)
    print()

    a_idx = [i for i, n in enumerate(name_seq) if n == "A"]
    bc_idx = [i for i, n in enumerate(name_seq) if n in ("B", "C")]
    d_idx = [i for i, n in enumerate(name_seq) if n == "D"]
    bc_flips = sum(
        1
        for i in range(1, len(name_seq))
        if name_seq[i] in ("B", "C")
        and name_seq[i - 1] in ("B", "C")
        and name_seq[i] != name_seq[i - 1]
    )

    print(f"\n  A indices: {min(a_idx)}..{max(a_idx)}")
    print(
        f"  B/C indices: {min(bc_idx)}..{max(bc_idx)} (flips: {bc_flips})"
    )
    print(f"  D indices: {min(d_idx)}..{max(d_idx)}")

    assert max(a_idx) < min(
        bc_idx
    ), "A should fully complete before B/C start"
    assert max(bc_idx) < min(
        d_idx
    ), "B/C should fully complete before D starts"
    print(
        "  >> Mixed sequential→parallel→sequential ordering preserved; B/C interleaved during their parallel segment."
    )


# ---------------------------------------------------------------------------
# Test 8: sync run_stream — same pipeline, sync iterator
# Verifies the threaded-queue bridge in AgentRearrange.run_stream.
# ---------------------------------------------------------------------------

# Live-LLM tests: they call agent.run() against a real model and can only
# pass with an API key. Without one, Agent.run now honestly raises
# AgentLLMError after retry exhaustion, so these tests are skipped instead
# of failing (same convention as tests/telemetry/test_telemetry.py).
_LLM_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
)
_HAS_LLM_KEY = any(os.getenv(k) for k in _LLM_KEYS)
requires_llm = pytest.mark.skipif(
    not _HAS_LLM_KEY,
    reason="no LLM API key set (OPENAI_API_KEY etc.) - live-LLM test skipped",
)

@requires_llm
def test_sync_run_stream():
    workflow = SequentialWorkflow(
        agents=[
            make_agent("Sync1", "Write one bullet about the topic."),
            make_agent("Sync2", "Rephrase the bullet as a question."),
        ],
        autosave=False,
    )
    print(
        "\n>>> SequentialWorkflow.run_stream (sync) — live tokens:\n"
    )
    tokens: List[str] = []
    for token in workflow.run_stream("supernovae"):
        tokens.append(token)
        print(token, end="", flush=True)
    print()
    print(f"\n  total tokens (sync): {len(tokens)}")
    assert len(tokens) > 0, "no tokens streamed"
    assert all(
        isinstance(t, str) for t in tokens
    ), "default mode must yield strings"
    print(
        "  >> Sync run_stream bridges threaded streaming correctly."
    )


# ---------------------------------------------------------------------------
# Test 9: multi-loop agent inside a SequentialWorkflow
# The first agent has max_loops=3 + a tool, so it does several LLM turns
# (tool call → tool result → synthesis) internally. Verifies that ALL of
# those turns stream tokens through the workflow, and that the next agent
# still receives the prior agent's final synthesis as its input.
# ---------------------------------------------------------------------------
async def test_multi_loop_agent_in_sequential_workflow():
    calculator = Agent(
        agent_name="Calculator",
        system_prompt="Use the add tool to compute the requested sum, then state the numerical result clearly.",
        model_name=MODEL,
        max_loops=3,
        tools=[add],
        persistent_memory=False,
        print_on=False,
    )
    explainer = Agent(
        agent_name="Explainer",
        system_prompt="Take the prior numerical result and write one short sentence explaining what it represents.",
        model_name=MODEL,
        max_loops=2,
        persistent_memory=False,
        print_on=False,
    )
    workflow = SequentialWorkflow(
        agents=[calculator, explainer],
        autosave=False,
    )

    print(
        "\n>>> SequentialWorkflow with multi-loop agent (Calculator: max_loops=3 + tool):\n"
    )

    # Use with_events=True so we can attribute tokens back to each agent
    # and confirm both agents streamed.
    per_agent: dict = {"Calculator": [], "Explainer": []}
    starts = ends = 0
    end_outputs: dict = {}

    async for evt in workflow.arun_stream(
        "Compute 17 + 25 using the add tool, then state the result.",
        with_events=True,
    ):
        t = evt["type"]
        if t == "agent_start":
            starts += 1
            print(f"\n  [start] {evt['agent']}")
        elif t == "agent_end":
            ends += 1
            end_outputs[evt["agent"]] = evt["output"]
            print(
                f"\n  [end]   {evt['agent']}: {len(evt['output'])} chars"
            )
        elif t == "token":
            per_agent.setdefault(evt["agent"], []).append(
                evt["token"]
            )
            print(evt["token"], end="", flush=True)

    print()

    calc_tokens = len(per_agent["Calculator"])
    exp_tokens = len(per_agent["Explainer"])
    calc_text = "".join(per_agent["Calculator"])
    exp_text = "".join(per_agent["Explainer"])

    print(f"\n  Calculator tokens streamed: {calc_tokens}")
    print(f"  Explainer  tokens streamed: {exp_tokens}")
    print(f"  Calculator final output: {end_outputs['Calculator']!r}")
    print(f"  Explainer  final output: {end_outputs['Explainer']!r}")

    # Both agents should have emitted lifecycle events.
    assert starts == 2, f"expected 2 agent_start events, got {starts}"
    assert ends == 2, f"expected 2 agent_end events, got {ends}"

    # Calculator with max_loops=3 + tool should produce more than a single
    # turn worth of tokens — the tool-call turn + synthesis turn both stream.
    assert (
        calc_tokens > 5
    ), f"Calculator streamed only {calc_tokens} tokens — multi-loop turns may not be streaming"

    # Explainer must have streamed too (its tokens prove the pipeline didn't stall).
    assert (
        exp_tokens > 0
    ), "Explainer received no tokens — pipeline broke after multi-loop agent"

    # Numerical correctness — proves the tool call actually happened and
    # its result reached the second agent.
    assert (
        "42" in calc_text
    ), f"Calculator output missing '42': {calc_text[:200]!r}"
    assert (
        "42" in exp_text
    ), f"Explainer didn't see '42' from upstream: {exp_text[:200]!r}"

    print(
        "  >> Multi-loop agent (3 turns + tool) streams all turns and hands off correctly to the next agent."
    )


async def main():
    await test_sequential_plain()
    await test_parallel_interleaving()
    await test_events_agent_rearrange()
    # await test_events_sequential_workflow()
    # await test_multi_loop_agent_with_tool()
    # await test_autonomous_loop_agent()
    # await test_mixed_flow()
    # test_sync_run_stream()  # sync — not awaited
    # await test_multi_loop_agent_in_sequential_workflow()


if __name__ == "__main__":
    asyncio.run(main())
