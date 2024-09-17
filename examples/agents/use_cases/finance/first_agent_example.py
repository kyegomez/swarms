import os
from swarms import Agent
from swarm_models import Anthropic
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)
from swarms.utils.data_to_text import data_to_text

# Initialize the agent
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    agent_description="Agent creates ",
    llm=Anthropic(anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")),
    max_loops="auto",
    autosave=True,
    # dynamic_temperature_enabled=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    # interactive=True, # Set to False to disable interactive mode
    dynamic_temperature_enabled=True,
    saved_state_path="finance_agent.json",
    # tools=[Add your functions here# ],
    # stopping_token="Stop!",
    # interactive=True,
    # docs_folder="docs", # Enter your folder name
    # pdf_path="docs/finance_agent.pdf",
    # sop="Calculate the profit for a company.",
    # sop_list=["Calculate the profit for a company."],
    user_name="swarms_corp",
    # # docs=
    # # docs_folder="docs",
    retry_attempts=3,
    # context_length=1000,
    # tool_schema = dict
    context_length=200000,
    # agent_ops_on=True,
    # long_term_memory=ChromaDB(docs_folder="artifacts"),
)


contract = data_to_text("your_contract_pdf.pdf")

agent.run(
    f"Analyze the following contract and give me a full summary: {contract}"
)
