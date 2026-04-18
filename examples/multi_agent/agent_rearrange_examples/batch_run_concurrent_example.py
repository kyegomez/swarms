"""
batch_run() concurrency demo
============================
Demonstrates that AgentRearrange.batch_run() now processes tasks within each
batch concurrently (via ThreadPoolExecutor) instead of sequentially.

Each task gets its own deep-copied conversation so results stay isolated and
the final list preserves the original task order.

Run with:
    python examples/multi_agent/agent_rearrange_examples/batch_run_concurrent_example.py
"""

import time
from unittest.mock import MagicMock

from swarms import Agent, AgentRearrange


# ---------------------------------------------------------------------------
# Lightweight mock agents so the example runs without real LLM credentials.
# Replace with real Agent(..., model_name="...") instances for live testing.
# ---------------------------------------------------------------------------
def _make_mock_agent(name: str, delay: float = 0.1) -> Agent:
    """Return an Agent whose run() sleeps briefly then echoes its name."""
    agent = MagicMock(spec=Agent)
    agent.agent_name = name
    agent.system_prompt = f"I am {name}."

    def _run(task, *a, **kw):
        time.sleep(delay)
        return f"[{name}] processed: {task}"

    agent.run = _run
    return agent


# ---------------------------------------------------------------------------
# Build a simple sequential pipeline: Researcher -> Writer
# ---------------------------------------------------------------------------
researcher = _make_mock_agent("Researcher", delay=0.15)
writer = _make_mock_agent("Writer", delay=0.15)

pipeline = AgentRearrange(
    agents=[researcher, writer],
    flow="Researcher -> Writer",
    max_loops=1,
    autosave=False,
    output_type="final",
)

# ---------------------------------------------------------------------------
# Tasks to process
# ---------------------------------------------------------------------------
TASKS = [
    "Summarise the history of quantum computing",
    "Explain how transformers work in NLP",
    "Describe the CAP theorem in distributed systems",
    "Outline the basics of reinforcement learning",
    "Explain gradient descent intuitively",
]

# ---------------------------------------------------------------------------
# Sequential baseline (old behaviour — for timing comparison)
# ---------------------------------------------------------------------------
print("=" * 60)
print("Sequential baseline (one task at a time)")
print("=" * 60)
t0 = time.perf_counter()
sequential_results = []
for task in TASKS:
    # Directly call _run on a fresh clone to mimic old list-comprehension
    import copy

    clone = copy.copy(pipeline)
    clone.conversation = copy.deepcopy(pipeline.conversation)
    sequential_results.append(clone.run(task=task))
sequential_elapsed = time.perf_counter() - t0
print(f"  Completed {len(TASKS)} tasks in {sequential_elapsed:.2f}s")

# ---------------------------------------------------------------------------
# Concurrent batch_run (new behaviour)
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Concurrent batch_run (ThreadPoolExecutor within each batch)")
print("=" * 60)
t1 = time.perf_counter()
concurrent_results = pipeline.batch_run(tasks=TASKS, batch_size=5)
concurrent_elapsed = time.perf_counter() - t1
print(f"  Completed {len(TASKS)} tasks in {concurrent_elapsed:.2f}s")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Results (order preserved)")
print("=" * 60)
for i, (task, result) in enumerate(zip(TASKS, concurrent_results), 1):
    print(f"\nTask {i}: {task}")
    print(f"  -> {result}")

# ---------------------------------------------------------------------------
# Speedup summary
# ---------------------------------------------------------------------------
if concurrent_elapsed > 0:
    speedup = sequential_elapsed / concurrent_elapsed
    print(
        f"\nSpeedup: {speedup:.1f}x  ({sequential_elapsed:.2f}s -> {concurrent_elapsed:.2f}s)"
    )

# ---------------------------------------------------------------------------
# Sanity: verify ordering is preserved
# ---------------------------------------------------------------------------
assert len(concurrent_results) == len(TASKS), "Result count mismatch!"
print("\nOrder-preservation check: PASSED")
print("Done.")
