"""
Example: Fully Save and Load an Agent (with Conversation History)

This demonstrates how to:
  1. Auto-save conversation messages to JSON
  2. Save the full Agent state
  3. Load both the Agent state and the conversation back into a fresh Agent
"""

import os
from swarms.structs.agent import Agent

# Helper to safely print type or None for agent properties
def print_agent_properties(agent, label):
    print(f"\n--- {label} ---")
    for prop in ["tokenizer", "long_term_memory", "logger_handler", "agent_output", "executor"]:
        value = getattr(agent, prop, None)
        print(f"{prop}: {type(value)}")

# Helper to extract the conversation history list
def get_conversation_history(agent):
    conv = getattr(agent, "conversation", None) or getattr(agent, "short_memory", None)
    return getattr(conv, "conversation_history", None)

# Robust helper to reload conversation from JSON into the correct attribute
def reload_conversation_from_json(agent, filepath):
    conv = getattr(agent, "conversation", None) or getattr(agent, "short_memory", None)
    if conv and hasattr(conv, "load_from_json"):
        conv.load_from_json(filepath)

# --- 1. Setup: Create and configure an agent with auto-save conversation ---
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
    conversation_kwargs={
        "auto_save": True,
        "save_as_json_bool": True,
        "save_filepath": "test_conversation_history.json"
    }
)

# --- 2. Interact to populate conversation ---
agent.run(task="hello")
agent.run(task="What is your purpose?")
agent.run(task="Tell me a joke.")
agent.run(task="Summarize our conversation so far.")

# --- 3. Inspect before saving ---
print_agent_properties(agent, "BEFORE SAVE")
print("\nConversation history BEFORE SAVE:", get_conversation_history(agent))

# --- 4. Save the agent state (conversation JSON was auto-saved under workspace) ---
state_path = os.path.join(agent.workspace_dir, "test_state.json")
agent.save(state_path)

# --- 5. Check that the conversation JSON file exists and print its contents ---
json_path = os.path.join(agent.workspace_dir, "test_conversation_history.json")
if os.path.exists(json_path):
    print(f"\n[CHECK] Conversation JSON file found: {json_path}")
    with open(json_path, "r") as f:
        json_data = f.read()
        print("[CHECK] JSON file contents:\n", json_data)
else:
    print(f"[WARN] Conversation JSON file not found: {json_path}")

# --- 6. Simulate fresh environment ---
del agent

# --- 7. Load: Restore the agent configuration ---
agent2 = Agent(agent_name="test")
agent2.load(state_path)

# --- 8. Load: Restore the conversation history from the workspace directory into a new Conversation object ---
from swarms.structs.conversation import Conversation
conversation_loaded = Conversation()
if os.path.exists(json_path):
    conversation_loaded.load_from_json(json_path)
    print("\n[CHECK] Loaded conversation from JSON into new Conversation object:")
    print(conversation_loaded.conversation_history)
else:
    print(f"[WARN] Conversation JSON file not found for loading: {json_path}")

# --- 9. Assign loaded conversation to agent2 and check ---
if hasattr(agent2, "conversation"):
    agent2.conversation = conversation_loaded
elif hasattr(agent2, "short_memory"):
    agent2.short_memory = conversation_loaded
print("\n[CHECK] Agent2 conversation history after assigning loaded conversation:", get_conversation_history(agent2))

# --- 10. Inspect after loading ---
print_agent_properties(agent2, "AFTER LOAD")
print("\nConversation history AFTER LOAD:", get_conversation_history(agent2))

# --- 11. Confirm the agent can continue ---
result = agent2.run(task="What is 2+2?")
print("\nAgent2 run result:", result)

# --- 12. Cleanup test files ---
for path in (state_path, json_path):
    try:
        os.remove(path)
    except OSError:
        pass
