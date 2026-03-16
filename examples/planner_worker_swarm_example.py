"""
PlannerWorkerSwarm Example
==========================

Demonstrates the planner-worker-judge architecture with parallel execution:

1. A **Planner** agent decomposes the goal into independent tasks
2. **Worker** agents claim tasks from a shared queue and execute them
   concurrently in a ThreadPoolExecutor — no worker-to-worker coordination
3. A **Judge** agent evaluates the combined results and decides:
   complete / fill gaps / fresh start (to combat drift)

Based on Cursor's "Scaling long-running autonomous coding" research.
https://cursor.com/blog/scaling-agents
"""

import time

from swarms import Agent
from swarms.structs.planner_worker_swarm import PlannerWorkerSwarm


def main():
    # --- Define 5 worker agents with different expertise ---
    # More workers = more parallelism. Each worker independently
    # claims tasks from the queue — they never talk to each other.

    workers = [
        Agent(
            agent_name="Research-Agent",
            agent_description="Gathers factual information and data points",
            system_prompt=(
                "You are a research specialist. Provide thorough, factual "
                "information with specific details. Be concise but comprehensive."
            ),
            model_name="gpt-4o-mini",
            max_loops=1,
        ),
        Agent(
            agent_name="Analysis-Agent",
            agent_description="Analyzes data and identifies patterns",
            system_prompt=(
                "You are an analysis specialist. Analyze information critically, "
                "identify patterns, and provide structured conclusions with bullet points."
            ),
            model_name="gpt-4o-mini",
            max_loops=1,
        ),
        Agent(
            agent_name="Writing-Agent",
            agent_description="Creates clear, well-structured content",
            system_prompt=(
                "You are a writing specialist. Produce clear, well-organized "
                "content with good readability and logical flow."
            ),
            model_name="gpt-4o-mini",
            max_loops=1,
        ),
        Agent(
            agent_name="Strategy-Agent",
            agent_description="Evaluates strategic implications and recommendations",
            system_prompt=(
                "You are a strategy specialist. Evaluate information from a "
                "strategic perspective and provide actionable recommendations."
            ),
            model_name="gpt-4o-mini",
            max_loops=1,
        ),
        Agent(
            agent_name="Data-Agent",
            agent_description="Compiles statistics, comparisons, and quantitative data",
            system_prompt=(
                "You are a data specialist. Compile relevant statistics, "
                "create comparisons, and present quantitative insights clearly."
            ),
            model_name="gpt-4o-mini",
            max_loops=1,
        ),
    ]

    # --- Create the swarm ---
    swarm = PlannerWorkerSwarm(
        name="Market-Research-Swarm",
        description="Conducts market research through parallel agent collaboration",
        agents=workers,
        max_loops=1,
        planner_model_name="gpt-4o-mini",
        judge_model_name="gpt-4o-mini",
        max_workers=5,  # All 5 workers can run concurrently
        worker_timeout=120,
        output_type="string",
    )

    # --- Run and time it ---
    task = (
        "Research the current state of the electric vehicle (EV) market. "
        "Cover: top manufacturers by market share, key technology trends, "
        "biggest challenges facing EV adoption, regional market differences, "
        "and a 5-year outlook. Each topic should be a separate task."
    )

    print(f"Starting swarm with {len(workers)} parallel workers...")
    print(f"Task: {task}\n")

    start = time.time()
    result = swarm.run(task=task)
    elapsed = time.time() - start

    # --- Display results ---
    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(result)

    # --- Show concurrency proof via task queue status ---
    print("\n" + "=" * 70)
    print("TASK QUEUE STATUS")
    print("=" * 70)
    status = swarm.get_status()
    for t in status["queue"]["tasks"]:
        print(
            f"  [{t['status']:>9}] {t['title'][:60]:<60} "
            f"-> {t['assigned_worker'] or 'unassigned'}"
        )

    print(f"\nTotal tasks: {status['queue']['total']}")
    print(f"Completed:   {status['queue']['progress']}")
    print(f"Wall time:   {elapsed:.1f}s (parallel execution)")
    print(
        f"Workers:     {len(workers)} concurrent threads\n"
        f"\nIf sequential, {status['queue']['total']} tasks at ~5-8s each "
        f"would take ~{status['queue']['total'] * 6}s.\n"
        f"Parallel wall time: {elapsed:.1f}s — that's the concurrency."
    )


if __name__ == "__main__":
    main()
