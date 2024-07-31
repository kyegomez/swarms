from swarms import Agent, OpenAIChat

## Initialize the workflow
agent = Agent(
    llm=OpenAIChat(),
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
)

# Run the workflow on a task
agent("Find a chick fil a equivalent in hayes valley")
