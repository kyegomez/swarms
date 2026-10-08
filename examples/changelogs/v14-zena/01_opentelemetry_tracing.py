"""Zena: OpenTelemetry tracing across Swarms.

Every agent run, swarm run, tool call, and LLM request emits a span. Spans
nest, so this two-agent workflow appears as one trace with two children.

Telemetry is ON by default. One switch turns it off.
"""

import os

# Opt out entirely: "false", "0", "no", "off", or an empty value.
# Comment this out to see spans emitted.
os.environ["SWARMS_TELEMETRY_ON"] = "false"

from swarms import Agent, SequentialWorkflow  # noqa: E402

researcher = Agent(
    agent_name="Researcher", model_name="gpt-5.4", max_loops=1
)
writer = Agent(agent_name="Writer", model_name="gpt-5.4", max_loops=1)

workflow = SequentialWorkflow(
    agents=[researcher, writer], max_loops=1
)
result = workflow.run(
    "Summarize the state of solid-state battery research."
)

print(result)
