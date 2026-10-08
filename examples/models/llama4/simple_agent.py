from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)
from dotenv import load_dotenv

load_dotenv()

# Initialize the agent
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    agent_description="Personal finance advisor agent",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    max_loops=1,
    model_name="groq/meta-llama/llama-4-scout-17b-16e-instruct",
    dynamic_temperature_enabled=True,
)

print(
    agent.run(
        "Perform a comprehensive analysis of the most promising undervalued ETFs, considering market trends, historical performance, and potential growth opportunities. Please think through the analysis for 2 internal loops to refine your insights."
    )
)
