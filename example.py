from swarms import Agent, Anthropic


# Initialize the agemt
agent = Agent(
    agent_name="Transcript Generator",
    agent_description=(
        "Generate a transcript for a youtube video on what swarms" " are!"
    ),
    llm=Anthropic(),
    max_loops=3,
    autosave=True,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    interactive=True,
)

# Run the Agent on a task
agent("Generate a transcript for a youtube video on what swarms are!")
