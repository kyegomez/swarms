from swarms import Agent

agent = Agent(
    agent_name="InteractiveAssistant",
    agent_description="An assistant you can chat with interactively in the terminal",
    model_name="gpt-5.4",
    max_loops=2,
    interactive=True,  # enables the interactive REPL
    persistent_memory=True,  # remembers conversation across sessions
    streaming_on=True,
)

print("=== Interactive mode (v12) ===")
print(
    "Type a message and press Enter. Press Ctrl+C or type 'exit' to quit.\n"
)

# run() with no task drops into the interactive REPL when interactive=True.
agent.run(task=None)
