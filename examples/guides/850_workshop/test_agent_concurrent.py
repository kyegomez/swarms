from swarms import Agent, run_agents_concurrently
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
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


outputs = run_agents_concurrently(
    agents=[equity_analyst_1, equity_analyst_2],
    task="What are the best ETFs for energy? Provide a reasoned comparison between top options and summarize your findings in a markdown table.",
    return_agent_output_dict=True,
)

print(outputs)
