from swarms import Agent, GroupChat

optimist = Agent(
    agent_name="Optimist",
    system_prompt="You argue for the benefits and upside of the topic.",
    model_name="openrouter/anthropic/claude-opus-4-8",
    max_loops=1,
    persistent_memory=False,
)

skeptic = Agent(
    agent_name="Skeptic",
    system_prompt="You argue for the risks and downsides of the topic.",
    model_name="openrouter/z-ai/glm-5.2",
    max_loops=1,
    persistent_memory=False,
)

realist = Agent(
    agent_name="Realist",
    system_prompt="You seek a balanced, evidence-based middle ground.",
    model_name="openrouter/tencent/hy3:free",
    max_loops=1,
    persistent_memory=False,
)

chat = GroupChat(
    agents=[optimist, skeptic, realist],
    max_loops=9,       # stop after 9 total messages
    threshold=0.5,     # only publish replies scoring above 0.5
)

result = chat.run(
    "Should hospitals adopt AI for first-pass medical diagnosis?"
)
print(result)