from swarms import Agent, AzureOpenAI

## Initialize the workflow
agent = Agent(
    llm=AzureOpenAI(),
    max_loops="auto",
    autosave=True,
    dashboard=False,
    streaming_on=True,
    verbose=True,
)

# Run the workflow on a task
agent("Understand the risk profile of this account")
