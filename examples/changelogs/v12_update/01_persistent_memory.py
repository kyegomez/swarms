from swarms import Agent

# --- Persistent agent (default behaviour) ---
# On first run it creates MEMORY.md. On subsequent runs it picks up
# where it left off — the model sees the prior conversation as a
# system preamble.
persistent_agent = Agent(
    agent_name="ResearchAssistant",
    agent_description="Remembers context across sessions",
    model_name="gpt-4.1",
    max_loops=1,
    persistent_memory=True,  # default — state survives restarts
)

print("=== Run 1: tell the agent something to remember ===")
out1 = persistent_agent.run(
    "My name is Kye. Remember that I am working on a paper about multi-agent systems."
)
print(out1)

print("\n=== Run 2: same agent instance — picks up prior context ===")
out2 = persistent_agent.run(
    "What is my name and what am I working on?"
)
print(out2)

# --- Stateless agent ---
# Every call starts from zero — nothing is written to or read from disk.
stateless_agent = Agent(
    agent_name="EphemeralAgent",
    agent_description="No memory between sessions",
    model_name="gpt-4.1",
    max_loops=1,
    persistent_memory=False,  # no disk activity at all
)

print("\n=== Stateless agent — has no memory of prior context ===")
out3 = stateless_agent.run(
    "What is my name and what am I working on?"
)
print(out3)
