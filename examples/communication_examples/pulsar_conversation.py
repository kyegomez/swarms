from swarms import Agent
from swarms.communication.pulsar_struct import PulsarConversation

# Configure a Pulsar-backed conversation store
conversation_store = PulsarConversation(
    pulsar_host="pulsar://localhost:6650",  # adjust to your broker
    topic="support_conversation",
    token_count=False,
)

# Create an agent that uses this persistent memory
agent = Agent(
    agent_name="SupportAgent",
    system_prompt="You are a helpful assistant.",
    model_name="gpt-4o-mini",
    long_term_memory=conversation_store,
    max_loops=1,
    autosave=True,
)

response = agent.run("What time is check-out?")
print(response)

# View the messages as stored in Pulsar
print(conversation_store.get_messages())
