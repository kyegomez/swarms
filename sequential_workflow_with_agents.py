from swarms import Agent, OpenAIChat, SequentialWorkflow

# Example usage
llm = OpenAIChat(
    temperature=0.5,
    max_tokens=3000,
)

# Initialize the Agent with the language agent
agent1 = Agent(
    agent_name="John the writer",
    llm=llm,
    max_loops=1,
    dashboard=False,
)


# Create another Agent for a different task
agent2 = Agent("Summarizer", llm=llm, max_loops=1, dashboard=False)


# Create the workflow
workflow = SequentialWorkflow(
    name="Blog Generation Workflow",
    description=(
        "Generate a youtube transcript on how to deploy agents into"
        " production"
    ),
    max_loops=1,
    autosave=True,
    dashboard=False,
    agents=[agent1, agent2],
)

# Run the workflow
workflow.run()
