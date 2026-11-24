"""A fan-out whose answers are posted in agent order once the step completes."""

from swarms import Agent, ConcurrentWorkflow, FileChannel

angles = ["technology", "regulation", "supply chain"]
agents = [
    Agent(
        agent_name=f"{angle.title()}-Analyst",
        system_prompt=f"Analyse the task from the {angle} angle only.",
        model_name="gpt-5.4",
        max_loops=1,
    )
    for angle in angles
]

channel = FileChannel("runs/concurrent.md")

workflow = ConcurrentWorkflow(agents=agents, channel=channel)
workflow.run(
    "What will shape the EV battery market over the next five years?"
)

print(channel.path.read_text())
