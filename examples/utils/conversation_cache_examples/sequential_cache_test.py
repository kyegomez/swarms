"""
Sequential workflow cache test.

Runs 3 agents in sequence and reports cache performance at two points:

  1. During execution  – each agent loop adds a message then reads the history
     once, so every read is a miss (cache is always stale when the loop starts).

  2. Post-execution reads – simulates realistic downstream usage such as
     logging, rendering, or exporting the history.  The history is stable at
     this point, so every read after the first is a hit.
"""

from swarms import Agent
from swarms.structs.sequential_workflow import SequentialWorkflow


def make_agent(name: str, role: str) -> Agent:
    agent = Agent(
        agent_name=name,
        system_prompt=role,
        model_name="claude-sonnet-4-5",
        max_loops=1,
        verbose=False,
        temperature=1.0,
    )
    agent.short_memory.caching = True
    return agent


def cache_stats(agent: Agent) -> dict:
    return agent.short_memory.get_cache_stats()


def print_stats(label: str, stats: dict) -> None:
    total = stats["hits"] + stats["misses"]
    print(
        f"  {label:<12} | calls={total:>3}  hits={stats['hits']:>3}"
        f"  misses={stats['misses']:>3}  hit_rate={stats['hit_rate']:>4.0%}"
        f"  tokens={stats['cached_tokens']:>5}"
    )


def main() -> None:
    researcher = make_agent(
        "Researcher",
        "You are a research agent. Give a concise 2-sentence summary of the topic.",
    )
    analyst = make_agent(
        "Analyst",
        "You are an analyst. List the 3 most important implications of the research.",
    )
    writer = make_agent(
        "Writer",
        "You are a writer. Combine the analysis into one clear, punchy paragraph.",
    )
    agents = [researcher, analyst, writer]

    workflow = SequentialWorkflow(
        name="CacheTestWorkflow",
        agents=agents,
        max_loops=1,
        verbose=False,
    )

    task = (
        "The impact of large language models on software development"
    )

    print("\n=== Sequential Workflow Cache Test ===")
    print(f"Task: {task}\n")

    workflow.run(task)

    # ── 1. Stats captured immediately after execution ─────────────────────────
    # Each agent loop follows: add(msg) → miss → LLM → add(response) → invalidate
    # So every read during execution is a miss; 0 hits is correct here.
    print("\n─── Stats after execution (reads during the run) ───")
    print(
        f"  {'Agent':<12} | {'calls':>6}  {'hits':>5}  {'misses':>7}  {'hit_rate':>9}  {'tokens':>7}"
    )
    for agent in agents:
        print_stats(agent.agent_name, cache_stats(agent))

    # ── 2. Post-execution reads on stable history ─────────────────────────────
    # After the workflow the history no longer changes.  Reading it multiple
    # times (e.g. for logging, export, display) returns the cached string.
    print(
        "\n─── Reading each agent's stable history 4× post-execution ───"
    )
    for agent in agents:
        for _ in range(4):
            agent.short_memory.return_history_as_string()

    print(
        f"\n  {'Agent':<12} | {'calls':>6}  {'hits':>5}  {'misses':>7}  {'hit_rate':>9}  {'tokens':>7}"
    )
    for agent in agents:
        print_stats(agent.agent_name, cache_stats(agent))

    # ── 3. Combined totals ─────────────────────────────────────────────────────
    all_stats = [cache_stats(a) for a in agents]
    total_hits = sum(s["hits"] for s in all_stats)
    total_misses = sum(s["misses"] for s in all_stats)
    total_calls = total_hits + total_misses
    overall_rate = (
        total_hits / total_calls if total_calls > 0 else 0.0
    )

    print("\n─── Combined totals ───")
    print(f"  Total calls:  {total_calls}")
    print(f"  Hits:         {total_hits}")
    print(f"  Misses:       {total_misses}")
    print(f"  Hit rate:     {overall_rate:.0%}\n")


if __name__ == "__main__":
    main()
