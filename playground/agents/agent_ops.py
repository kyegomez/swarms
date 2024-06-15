from swarms import Agent, OpenAIChat

# Initialize the agent
agent = Agent(
    agent_name="Accounting Agent",
    system_prompt="Generate a financial report for the company's quarterly earnings.",
    agent_description=(
        "Generate a financial report for the company's quarterly earnings."
    ),
    llm=OpenAIChat(),
    max_loops=1,
    autosave=True,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    interactive=False,
    state_save_file_type="json",
    saved_state_path="accounting_agent.json",
    agent_ops_on=True,
)

# Run the Agent on a task
agent.run(
    "Generate a financial report for the company's quarterly earnings!"
)
