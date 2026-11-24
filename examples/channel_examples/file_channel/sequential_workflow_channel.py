"""A pipeline whose whole transcript lands in one markdown file."""

from swarms import Agent, FileChannel, SequentialWorkflow

researcher = Agent(
    agent_name="Researcher",
    system_prompt="Gather the key facts as a short bullet list.",
    model_name="gpt-5.4",
    max_loops=1,
)
writer = Agent(
    agent_name="Writer",
    system_prompt="Turn the facts in the conversation into a short report.",
    model_name="gpt-5.4",
    max_loops=1,
)
reviewer = Agent(
    agent_name="Reviewer",
    system_prompt="Point out anything wrong or missing in the report.",
    model_name="gpt-5.4",
    max_loops=1,
)

channel = FileChannel("runs/sequential.md")

workflow = SequentialWorkflow(
    agents=[researcher, writer, reviewer],
    max_loops=1,
    channel=channel,
)
workflow.run("Report on the EV battery market.")

print(channel.path.read_text())
