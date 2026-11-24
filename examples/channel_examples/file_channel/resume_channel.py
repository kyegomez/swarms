"""A second run continues an existing file: the earlier turns become prior context.

Run sequential_workflow_channel.py first, then this. Edit runs/sequential.md
between the two to steer the follow-up.
"""

from swarms import Agent, FileChannel, SequentialWorkflow

editor = Agent(
    agent_name="Editor",
    system_prompt="Tighten the report in the conversation to half its length.",
    model_name="gpt-5.4",
    max_loops=1,
)

channel = FileChannel("runs/sequential.md")

SequentialWorkflow(agents=[editor], max_loops=1, channel=channel).run(
    "Shorten the report above."
)

print(channel.path.read_text())
