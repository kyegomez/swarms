import os
import tempfile
from swarms.structs.conversation import Conversation

# Use a temp directory so this example is self-contained and repeatable.
workspace = tempfile.mkdtemp(prefix="swarms_v12_conv_")

print(f"Workspace: {workspace}")

memory_path = os.path.join(
    workspace, "agents", "DemoAgent", "MEMORY.md"
)

# --- Create a conversation and add some messages ---
conv = Conversation(
    system_prompt="You are a helpful assistant.",
    memory_md_path=memory_path,
    time_enabled=True,
)

conv.add("user", "What is the capital of France?")
conv.add("assistant", "The capital of France is Paris.")
conv.add("user", "And what is the population of Paris?")
conv.add(
    "assistant",
    "The population of the Paris metropolitan area is roughly 12 million.",
)

print("\n=== History with ISO timestamps ===")
print(conv.return_history_as_string())

# Confirm MEMORY.md was written
if os.path.exists(memory_path):
    print(f"\n=== MEMORY.md exists at {memory_path} ===")
    with open(memory_path) as f:
        print(f.read()[:400])

# --- Reload the conversation from disk ---
# A brand-new Conversation with the same memory_md_path reads MEMORY.md
# as a preamble — state persists across instantiations.
conv2 = Conversation(
    system_prompt="You are a helpful assistant.",
    memory_md_path=memory_path,
)
print("\n=== Reloaded conversation history ===")
print(conv2.return_history_as_string()[:400])

# --- Compact the conversation ---
# Replaces the raw interaction history with a single summary.
# A timestamped archive of the full history is created before the rewrite.
summary_text = (
    "The user asked about France: capital is Paris, "
    "metropolitan population ~12 million."
)
conv.compact(summary=summary_text, summary_role="System")

print("\n=== History after compact ===")
print(conv.return_history_as_string())

# Check for archive files
archive_dir = os.path.join(workspace, "agents", "DemoAgent")
archives = [
    f
    for f in os.listdir(archive_dir)
    if "archive" in f.lower() or f.endswith(".md")
]
print(f"\nFiles in agent dir after compact: {archives}")
