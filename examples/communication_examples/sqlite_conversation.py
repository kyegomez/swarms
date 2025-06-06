from swarms import Agent
from swarms.communication.sqlite_wrap import SQLiteConversation

# Configure a conversation store backed by SQLite
conversation_store = SQLiteConversation(
    db_path="agent_history.db",
    table_name="messages",
    enable_logging=True,
)

# Create an agent that leverages the SQLite-based memory
agent = Agent(
    agent_name="SupportAgent",
    system_prompt="You are a helpful assistant.",
    model_name="gpt-4o-mini",
    long_term_memory=conversation_store,
    max_loops=1,
    autosave=True,
)

response = agent.run("How do I reset my password?")
print(response)

# Show the conversation as stored in SQLite
print(conversation_store.to_dict())
