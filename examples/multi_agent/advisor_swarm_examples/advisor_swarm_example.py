"""
AdvisorSwarm Example

The advisor provides strategic guidance; the executor does the work.
Swap in any models you want — it's provider-agnostic.

Loads API keys from .env automatically.
"""

from swarms import AdvisorSwarm

swarm = AdvisorSwarm(
    name="Code Advisor",
    description="Advisor-guided code generation",
    executor_model_name="claude-sonnet-4-6",
    advisor_model_name="claude-opus-4-6",
    max_advisor_uses=3,  # 1 plan + up to 2 review-refine cycles
    max_loops=1,
    verbose=True,
)

result = swarm.run(
    task=(
        "Write a Python function that implements binary search on a sorted list. "
        "Include proper error handling, type hints, and edge cases."
    )
)

print(result)
