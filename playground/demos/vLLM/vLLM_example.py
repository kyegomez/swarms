import os
import asyncio
from swarms import Agent, OpenAIChat
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)

# Set the OpenAI environment to use vLLM
api_key = os.getenv("OPENAI_API_KEY") or "EMPTY" # for vllm
api_base = os.getenv("OPENAI_API_BASE") or "http://localhost:8000/v1" # for vllm

# Create an instance of the OpenAIChat class
model = OpenAIChat(
    base_url=api_base, api_key=api_key, model="NousResearch/Meta-Llama-3-8B-Instruct", temperature=0.5, streaming=True, verbose=True
)

# Initialize the agent
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    llm=model,
    max_loops=2,
    autosave=True,
    # dynamic_temperature_enabled=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    # interactive=True, # Set to False to disable interactive mode
    dynamic_temperature_enabled=False,
    saved_state_path="finance_agent.json",
    # tools=[#Add your functions here# ],
    # stopping_token="Stop!",
    # interactive=True,
    # docs_folder="docs", # Enter your folder name
    # pdf_path="docs/finance_agent.pdf",
    # sop="Calculate the profit for a company.",
    # sop_list=["Calculate the profit for a company."],
    user_name="RAH@EntangleIT.com",
    # # docs=
    # # docs_folder="docs",
    retry_attempts=3,
    # context_length=1000,
    # tool_schema = dict
    context_length=200000,
    # tool_schema=
    # tools
    # agent_ops_on=True,
)

async def startup_event():
    agent.stream_reponse(
        "What are the components of a startups stock incentive equity plan"
    )

if __name__ == "__main__":
    asyncio.run(startup_event())