from swarms import Agent, SequentialWorkflow

agents = [
    Agent(agent_name="Researcher", model_name="gpt-5.4"),
    Agent(agent_name="Writer", model_name="claude-sonnet-4-6"),
    Agent(agent_name="Editor", model_name="claude-sonnet-4-6"),
]

workflow = SequentialWorkflow(
    agents=agents,
    drift_detection=True,  # enable post-pipeline alignment check
    drift_threshold=0.75,  # warn if alignment score < 0.75
    drift_model="claude-sonnet-4-5",
)

result = workflow.run(
    "Write a technical blog post about transformer attention mechanisms"
)
# If the Editor drifts too far from the original task, a warning is emitted
print(result)
