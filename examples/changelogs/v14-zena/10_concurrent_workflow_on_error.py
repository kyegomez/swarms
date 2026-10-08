"""Zena: explicit failure policy for ConcurrentWorkflow.

Previously one agent raising could abort the whole run and discard work
every other agent had already finished. on_error makes the policy explicit
and validates it at construction.

  "store" -> record the error as that agent's output, let the others finish
  "raise" -> propagate and abort the run

max_workers is sized for network-bound work, not CPU count. When omitted it
defaults to len(agents) capped at 32.
"""

from swarms import Agent, ConcurrentWorkflow

agents = [
    Agent(agent_name=f"Worker-{i}", model_name="gpt-5.4", max_loops=1)
    for i in range(5)
]

workflow = ConcurrentWorkflow(
    agents=agents,
    on_error="store",
    max_workers=8,
)

print(workflow.run("List ten use cases for multi-agent AI systems."))
