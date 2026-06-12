from swarms import Agent

# --- Agent with context compression enabled (default) ---
# After crossing ~90 % of context_length the compressor auto-summarises
# and rewrites MEMORY.md, keeping the window healthy.
agent_compressed = Agent(
    agent_name="LongRunAgent",
    agent_description="Handles very long sessions without hitting the context limit",
    model_name="gpt-4.1",
    max_loops=3,
    context_length=8000,  # token budget
    context_compression=True,  # default — auto-compress when near limit
    persistent_memory=True,
)

print("=== Agent with context compression ===")
result = agent_compressed.run(
    "Summarise the history of artificial intelligence in 10 detailed paragraphs, "
    "then list the 20 most important researchers and their contributions."
)
print(result)

# --- Agent with context compression disabled ---
# Useful for short, predictable tasks where you want raw fidelity and
# no background summarisation.
agent_raw = Agent(
    agent_name="ShortTaskAgent",
    agent_description="Single-shot, no compression overhead",
    model_name="gpt-4.1",
    max_loops=1,
    context_length=8000,
    context_compression=False,  # disabled — raw context only
    persistent_memory=False,
)

print("\n=== Agent without context compression ===")
result2 = agent_raw.run("What is 2 + 2?")
print(result2)
