from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)
from swarms.structs.multi_agent_exec import (
    run_agents_concurrently_uvloop,
)

# Initialize the equity analyst agents
equity_analyst_1 = Agent(
    agent_name="Equity-Analyst-1",
    agent_description="Equity research analyst focused on fundamental analysis",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    max_loops=1,
    model_name="gpt-4.1",
    dynamic_temperature_enabled=True,
)

equity_analyst_2 = Agent(
    agent_name="Equity-Analyst-2",
    agent_description="Equity research analyst focused on technical analysis",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    max_loops=1,
    model_name="gpt-4.1",
    dynamic_temperature_enabled=True,
)


outputs = run_agents_concurrently_uvloop(
    agents=[equity_analyst_1, equity_analyst_2],
    task="Analyze high growth tech stocks focusing on fundamentals like revenue growth, margins, and market position. Create a detailed analysis table in markdown.",
)

print(outputs)
