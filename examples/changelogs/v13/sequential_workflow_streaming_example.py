"""SequentialWorkflow streaming.

run_stream yields tokens from each agent in sequence as they are
generated. Pass with_events=True to get structured event dicts
(agent_start / token / agent_end) instead of plain strings — useful for
UIs that render per-agent panels. AgentRearrange and HierarchicalSwarm
gained the same run_stream / arun_stream pair.
"""

from swarms import Agent, SequentialWorkflow

pipeline = SequentialWorkflow(
    agents=[
        Agent(agent_name="Researcher", model_name="gpt-4.1", max_loops=1),
        Agent(agent_name="Writer", model_name="gpt-4.1", max_loops=1),
    ],
)

# Plain token stream
for token in pipeline.run_stream("Summarise LLM research this year."):
    print(token, end="", flush=True)
print()

# Structured events (for per-agent UI panels)
for event in pipeline.run_stream(
    "Summarise LLM research this year.", with_events=True
):
    if event["type"] == "agent_start":
        print(f"\n--- {event['agent']} started ---")
    elif event["type"] == "token":
        print(event["token"], end="", flush=True)
    elif event["type"] == "agent_end":
        print(f"\n--- {event['agent']} finished ---")
