"""
Example: Fully Save and Load an Agent (Issue #640)

This example demonstrates how to save and load an Agent instance such that all non-serializable properties
(tokenizer, long_term_memory, logger_handler, agent_output, executor) are restored after loading.

This is a user-facing, production-grade demonstration for swarms.
"""

from swarms.structs.agent import Agent
import os

# Helper to safely print type or None for agent properties
def print_agent_properties(agent, label):
    print(f"\n--- {label} ---")
    for prop in ["tokenizer", "long_term_memory", "logger_handler", "agent_output", "executor"]:
        value = getattr(agent, prop, None)
        print(f"{prop}: {type(value)}")

# --- Setup: Create and configure an agent ---
agent = Agent(
    agent_name="test",
    user_name="test_user",
    system_prompt="This is a test agent",
    max_loops=1,
    context_length=200000,
    autosave=True,
    verbose=True,
    artifacts_on=True,
    artifacts_output_path="test",
    artifacts_file_extension=".txt",
)

# Optionally, interact with the agent to populate state
agent.run(task="hello")

# Print non-serializable properties BEFORE saving
print_agent_properties(agent, "BEFORE SAVE")

# Save the agent state
save_path = os.path.join(agent.workspace_dir, "test_state.json")
agent.save(save_path)

# Delete the agent instance to simulate a fresh load
del agent

# --- Load: Restore the agent from file ---
agent2 = Agent(agent_name="test")  # Minimal init, will be overwritten by load
agent2.load(save_path)

# Print non-serializable properties AFTER loading
print_agent_properties(agent2, "AFTER LOAD")

# Confirm agent2 can still run tasks and autosave
result = agent2.run(task="What is 2+2?")
print("\nAgent2 run result:", result)

# Clean up test file
try:
    os.remove(save_path)
except Exception:
    pass
