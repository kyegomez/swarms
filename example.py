from swarms import Agent, Anthropic


## Initialize the workflow
agent = Agent(
    agent_name="Transcript Generator",
    agent_description=(
        "Generate a transcript for a youtube video on what swarms"
        " are!"
    ),
    llm=Anthropic(),
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
