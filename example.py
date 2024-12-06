import os

from dotenv import load_dotenv
from swarm_models import OpenAIChat

from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)
from swarms.structs.orchestrator import Orchestrator
from swarms.structs.notification_manager import UpdateMetadata
from datetime import datetime

load_dotenv()

# Get the OpenAI API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Create an instance of the OpenAIChat class
model = OpenAIChat(
    openai_api_key=api_key, model_name="gpt-4o-mini", temperature=0.1
)

# Initialize the agent
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    llm=model,
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="finance_agent.json",
    user_name="swarms_corp",
    retry_attempts=1,
    streaming_on=True,
    context_length=200000,
    return_step_meta=True,
    output_type="json",  # "json", "dict", "csv" OR "string" soon "yaml" and
    auto_generate_prompt=False,  # Auto generate prompt for the agent based on name, description, and system prompt, task
    artifacts_on=True,
    artifacts_output_path="roth_ira_report",
    artifacts_file_extension=".txt",
    max_tokens=8000,
    return_history=True,
)

# Create orchestrator
orchestrator = Orchestrator()

# Register agents
orchestrator.register_agent(agent)

# Example vector DB update
update = UpdateMetadata(
    topic="stock_market",
    importance=0.8,
    timestamp=datetime.now(),
    affected_areas=["finance", "trading"]
)

# Handle update - only Financial-Analysis-Agent will be notified
orchestrator.handle_vector_db_update(update)

agent.run(
    "How can I establish a ROTH IRA to buy stocks and get a tax break? What are the criteria. Create a report on this question.",
    all_cores=True,
)
