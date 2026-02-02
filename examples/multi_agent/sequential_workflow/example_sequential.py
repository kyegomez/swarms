"""
Sequential Multi-Agent Example

This example demonstrates sequential execution using SequentialWorkflow,
where agents work one after another and each agent builds on the previous output.

Use Case: Content creation pipeline where research, writing, and editing run in sequence.
"""

from swarms import Agent
from swarms.structs import SequentialWorkflow

# Create specialized agents for the sequential workflow
research_agent = Agent(
    agent_name="Research-Agent",
    agent_description="Expert at researching topics and gathering information",
    model_name="gpt-4o-mini",
    max_loops=1,
    system_prompt="You are a research specialist. Gather comprehensive information on the given topic and provide key findings.",
)

writer_agent = Agent(
    agent_name="Writer-Agent",
    agent_description="Expert at writing engaging content based on research",
    model_name="gpt-4o-mini",
    max_loops=1,
    system_prompt="You are a professional writer. Transform research findings into engaging, well-structured content.",
)

editor_agent = Agent(
    agent_name="Editor-Agent",
    agent_description="Expert at editing and refining content for quality",
    model_name="gpt-4o-mini",
    max_loops=1,
    system_prompt="You are an experienced editor. Review and improve the content for clarity, grammar, and impact.",
)

# Build the sequential workflow (research -> write -> edit)
workflow = SequentialWorkflow(
    agents=[research_agent, writer_agent, editor_agent],
    name="Content-Creation-Workflow",
    description="Sequential workflow: research, then write, then edit.",
    autosave=True,
    verbose=False,
)


out = workflow.run(
    "The history and future of artificial intelligence"
)
print(out)
