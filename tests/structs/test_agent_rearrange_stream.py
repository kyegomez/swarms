"""
Tests for AgentRearrange stream_between_nodes pipeline feature.

All tests use real Agent objects backed by a live LLM (gpt-5.4).
No mocks are used for the agents themselves; the pipeline mechanics
are exercised end-to-end through arun_stream / run_stream.
"""

import asyncio
import pytest
from swarms import Agent
from swarms.structs.agent_rearrange import AgentRearrange


def make_agent(name: str, description: str) -> Agent:
    return Agent(
        agent_name=name,
        agent_description=description,
        model_name="claude-sonnet-4-5",
        max_loops=1,
        verbose=False,
        print_on=False,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def collect_stream(rearrange: AgentRearrange, task: str):
    """Drain run_stream() and return all (agent_name, token) tuples."""
    return list(rearrange.run_stream(task=task))


async def collect_astream(rearrange: AgentRearrange, task: str):
    """Drain arun_stream() and return all (agent_name, token) tuples."""
    results = []
    async for item in rearrange.arun_stream(task=task):
        results.append(item)
    return results


# ---------------------------------------------------------------------------
# Basic pipeline tests
# ---------------------------------------------------------------------------


def test_stream_between_nodes_line_strategy_produces_tokens():
    """stream_between_nodes=True with line strategy yields tokens from all agents."""
    a = make_agent(
        "Writer", "Write a short two-sentence paragraph on AI."
    )
    b = make_agent("Editor", "Edit the paragraph for clarity.")

    rearrange = AgentRearrange(
        agents=[a, b],
        flow="Writer -> Editor",
        stream_between_nodes=True,
        buffer_strategy="line",
    )

    tokens = collect_stream(rearrange, task="Tell me about AI.")

    assert len(tokens) > 0, "Should produce tokens"
    agent_names = {t[0] for t in tokens}
    assert "Writer" in agent_names, "Writer should produce tokens"
    assert "Editor" in agent_names, "Editor should produce tokens"


def test_stream_between_nodes_tokens_strategy_produces_tokens():
    """stream_between_nodes=True with tokens strategy yields tokens from all agents."""
    a = make_agent("Researcher", "Summarise AI in three sentences.")
    b = make_agent(
        "Summarizer", "Condense the summary to one sentence."
    )

    rearrange = AgentRearrange(
        agents=[a, b],
        flow="Researcher -> Summarizer",
        stream_between_nodes=True,
        buffer_strategy="tokens",
        buffer_token_count=10,
    )

    tokens = collect_stream(rearrange, task="What is AI?")

    assert len(tokens) > 0
    agent_names = {t[0] for t in tokens}
    assert "Researcher" in agent_names
    assert "Summarizer" in agent_names


def test_stream_between_nodes_all_strategy_equivalent_to_no_pipeline():
    """buffer_strategy='all' produces same agent names as standard streaming."""
    a = make_agent("AgentA", "Answer in one sentence.")
    b = make_agent("AgentB", "Rephrase the answer.")

    rearrange_pipeline = AgentRearrange(
        agents=[a, b],
        flow="AgentA -> AgentB",
        stream_between_nodes=True,
        buffer_strategy="all",
    )
    rearrange_standard = AgentRearrange(
        agents=[a, b],
        flow="AgentA -> AgentB",
        stream_between_nodes=False,
    )

    tokens_pipeline = collect_stream(
        rearrange_pipeline, task="What is 2+2?"
    )
    tokens_standard = collect_stream(
        rearrange_standard, task="What is 2+2?"
    )

    names_pipeline = {t[0] for t in tokens_pipeline}
    names_standard = {t[0] for t in tokens_standard}
    assert names_pipeline == names_standard


def test_stream_between_nodes_four_agent_chain():
    """A 4-agent chain all produce tokens in pipeline mode."""
    agents = [
        make_agent("Step1", "Write one sentence about space."),
        make_agent("Step2", "Expand it to two sentences."),
        make_agent("Step3", "Add a fun fact."),
        make_agent("Step4", "Summarise everything in one sentence."),
    ]

    rearrange = AgentRearrange(
        agents=agents,
        flow="Step1 -> Step2 -> Step3 -> Step4",
        stream_between_nodes=True,
        buffer_strategy="line",
    )

    tokens = collect_stream(rearrange, task="Tell me about space.")

    agent_names = {t[0] for t in tokens}
    assert (
        len(agent_names) == 4
    ), f"Expected 4 distinct agents, got {agent_names}"


# ---------------------------------------------------------------------------
# with_events mode
# ---------------------------------------------------------------------------


def test_stream_between_nodes_with_events_structure():
    """with_events=True yields structured dicts with correct keys."""
    a = make_agent("Alpha", "Answer in one sentence.")
    b = make_agent("Beta", "Rephrase the answer.")

    rearrange = AgentRearrange(
        agents=[a, b],
        flow="Alpha -> Beta",
        stream_between_nodes=True,
        buffer_strategy="line",
    )

    events = list(
        rearrange.run_stream(
            task="What is the sun?", with_events=True
        )
    )

    event_types = {e["type"] for e in events if isinstance(e, dict)}
    assert "agent_start" in event_types
    assert "token" in event_types
    assert "agent_end" in event_types


# ---------------------------------------------------------------------------
# Backward compatibility: stream_between_nodes=False
# ---------------------------------------------------------------------------


def test_stream_between_nodes_false_standard_output():
    """stream_between_nodes=False (default) produces the same output as before."""
    a = make_agent("P", "Answer in one sentence.")
    b = make_agent("Q", "Rephrase the answer.")

    rearrange = AgentRearrange(
        agents=[a, b],
        flow="P -> Q",
        stream_between_nodes=False,
    )

    tokens = collect_stream(rearrange, task="What is Python?")
    agent_names = {t[0] for t in tokens}
    assert "P" in agent_names
    assert "Q" in agent_names


# ---------------------------------------------------------------------------
# Async interface
# ---------------------------------------------------------------------------


def test_arun_stream_pipeline_async():
    """arun_stream with stream_between_nodes=True works in an async context."""
    a = make_agent("AAsync", "Answer in one sentence.")
    b = make_agent("BAsync", "Expand the answer.")

    rearrange = AgentRearrange(
        agents=[a, b],
        flow="AAsync -> BAsync",
        stream_between_nodes=True,
        buffer_strategy="tokens",
        buffer_token_count=5,
    )

    tokens = asyncio.run(
        collect_astream(rearrange, task="What is Python?")
    )

    assert len(tokens) > 0
    agent_names = {t[0] for t in tokens}
    assert "AAsync" in agent_names
    assert "BAsync" in agent_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
