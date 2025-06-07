from swarms import Agent
from swarms.communication.duckdb_wrap import DuckDBConversation

# Configure a DuckDB-backed conversation store
conversation_store = DuckDBConversation(
    db_path="support_conversation.duckdb",
    table_name="support_history",
    enable_logging=True,
)

# Create an agent that uses this persistent memory
agent = Agent(
    agent_name="HelpdeskAgent",
    system_prompt="You are a helpful assistant.",
    model_name="gpt-4o-mini",
    long_term_memory=conversation_store,
    max_loops=1,
    autosave=False,
)

response = agent.run("What are your hours of operation?")
print(response)

# View the conversation as stored in DuckDB
print(conversation_store.to_dict())
