import os

from swarms_memory import ChromaDB

from swarms import Agent, Anthropic
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)

# Initilaize the chromadb client
chromadb = ChromaDB(
    metric="cosine",
    output_dir="fiance_agent_rag",
    # docs_folder="artifacts", # Folder of your documents
)

# Model
model = Anthropic(anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))


# Initialize the agent
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    agent_description="Agent creates ",
    llm=model,
    max_loops="auto",
    autosave=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    dynamic_temperature_enabled=True,
    saved_state_path="finance_agent.json",
    user_name="swarms_corp",
    retry_attempts=3,
    context_length=200000,
    long_term_memory=chromadb,
)


agent.run(
    "What are the components of a startups stock incentive equity plan"
)
