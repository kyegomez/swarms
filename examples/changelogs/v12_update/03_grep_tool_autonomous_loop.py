import os
from swarms import Agent

# Point the agent at a real directory it can grep through.
# Adjust this to a directory that exists on your machine.
TARGET_DIR = os.path.expanduser(
    "~/Desktop/research/swarms/swarms/structs"
)

agent = Agent(
    agent_name="CodeSearcher",
    agent_description="Finds patterns across source files using the built-in grep tool",
    model_name="gpt-4.1",
    max_loops="auto",  # autonomous loop — grep tool is available here
    interactive=False,
    persistent_memory=False,
)

print("=== Grep for all TODO comments in the structs package ===")
result = agent.run(
    f"Use the grep tool to find all TODO comments (case-insensitive) "
    f"in Python files under {TARGET_DIR}. "
    f"List each file and line number, then summarise the categories of work outstanding."
)
print(result)

print("\n=== Grep for a specific class definition ===")
result2 = agent.run(
    f"Search {TARGET_DIR} recursively for the definition of class Agent. "
    f"Show the file path and line number where it is defined."
)
print(result2)
