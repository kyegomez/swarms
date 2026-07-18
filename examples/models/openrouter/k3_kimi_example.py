from swarms import Agent

# Define a general system prompt for a versatile assistant
system_prompt = (
    "You are Kimi, a highly capable and helpful AI assistant. "
    "Your goal is to support users in a wide array of tasks, providing clear, insightful, and accurate information or recommendations. "
    "Communicate in a friendly, professional manner. "
    "Adapt your explanations to suit both beginners and experts. "
    "Where useful, provide reasoning for your answers, clarify any assumptions, and specify limitations if relevant. "
    "You are capable of helping with any subject or challenge a user brings."
)

# Initialize the general assistant agent
agent = Agent(
    agent_name="Kimi",
    agent_description="A general-purpose assistant for any task, providing helpful and thoughtful answers.",
    system_prompt=system_prompt,
    model_name="openrouter/moonshotai/kimi-k3",
    max_loops=1,
    top_p=None,
    temperature=None,
    thinking_tokens=1024,
    reasoning_effort="high",
    persistent_memory=False,
)

out = agent.run(
    task="How can I boost my productivity while working from home?",
)

print(out)
