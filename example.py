from swarms import Agent, OpenAIChat

## Initialize the workflow
agent = Agent(
    llm=OpenAIChat(),
    max_loops="auto",
    autosave=True,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
)

# Run the workflow on a task
agent(
    "Generate a transcript for a youtube video on what swarms are!"
    " Output a <DONE> token when done."
)
