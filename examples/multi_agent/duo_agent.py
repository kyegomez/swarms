from swarms import Agent
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
    user_name="swarms_corp",
    retry_attempts=3,
    context_length=8192,
    return_step_meta=False,
    output_type="str",
    auto_generate_prompt=False,
    max_tokens=4000,
    saved_state_path="equity_analyst_1.json",
    interactive=False,
    roles="analyst",
)

equity_analyst_2 = Agent(
    agent_name="Equity-Analyst-2",
    agent_description="Equity research analyst focused on technical analysis",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    max_loops=1,
    model_name="gpt-4.1",
    dynamic_temperature_enabled=True,
    user_name="swarms_corp",
    retry_attempts=3,
    context_length=8192,
    return_step_meta=False,
    output_type="str",
    auto_generate_prompt=False,
    max_tokens=4000,
    saved_state_path="equity_analyst_2.json",
    interactive=False,
    roles="analyst",
)

# Run analysis with both analysts
equity_analyst_1.talk_to(
    equity_analyst_2,
    "Analyze high growth tech stocks focusing on fundamentals like revenue growth, margins, and market position. Create a detailed analysis table in markdown.",
)
