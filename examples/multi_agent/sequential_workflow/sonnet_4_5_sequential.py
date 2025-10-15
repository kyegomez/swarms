from swarms import Agent, SequentialWorkflow

# Agent 1: The Researcher
researcher = Agent(
    agent_name="Researcher",
    system_prompt="Your job is to research the provided topic and provide a detailed summary.",
    model_name="anthropic/claude-sonnet-4-5",
    top_p=None,
    dynamic_temperature_enabled=True,
)

# Agent 2: The Writer
writer = Agent(
    agent_name="Writer",
    system_prompt="Your job is to take the research summary and write a beautiful, engaging blog post about it.",
    model_name="anthropic/claude-sonnet-4-5",
    top_p=None,
    dynamic_temperature_enabled=True,
)

# Create a sequential workflow where the researcher's output feeds into the writer's input
workflow = SequentialWorkflow(agents=[researcher, writer])

# Run the workflow on a task
final_post = workflow.run(
    "Create a comprehensive and detailed report on Gold ETFs"
)
print(final_post)
