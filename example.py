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
    state_save_file_type="json",
    saved_state_path="transcript_generator.json",
)

# Run the Agent on a task
out = agent.run("Generate a transcript for a youtube video on what swarms are!")
print(out)

# Save the state
check = agent.save_state(
    "transcript_generator.json",
    "Generate a transcript for a youtube video on what swarms are!",
)
