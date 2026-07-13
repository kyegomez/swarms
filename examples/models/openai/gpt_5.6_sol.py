from swarms import Agent

# Initialize the agent
agent = Agent(
    agent_name="Generalist-Assistant",
    agent_description="A versatile assistant capable of answering a wide range of questions and tackling diverse tasks.",
    system_prompt="You are a friendly and knowledgeable generalist assistant ready to help with any query or problem.",
    model_name="gpt-5.6-sol",
    max_loops=1,
    top_p=None,
    temperature=None,
    thinking_tokens=1024,
    reasoning_effort=None,
    persistent_memory=False,
)

out = agent.run(
    task="How can I become more productive in my daily life? Please provide practical tips and strategies.",
)

print(out)
