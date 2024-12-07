from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)

# Initialize the agent
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    agent_description = "Personal finance advisor agent",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT + "Output the <DONE> token when you're done creating a portfolio of etfs, index, funds, and more for AI",
    model_name="gpt-4o", # Use any model from litellm
    max_loops="auto",
    dynamic_temperature_enabled=True,
    user_name="Kye",
    retry_attempts=3,
    streaming_on=True,
    context_length=16000,
    return_step_meta=False,
    output_type="str",  # "json", "dict", "csv" OR "string" "yaml" and
    auto_generate_prompt=False,  # Auto generate prompt for the agent based on name, description, and system prompt, task
    max_tokens=16000, # max output tokens
    interactive = True,
    stopping_token="<DONE>",
    execute_tool=True,
)

agent.run(
    "Create a table of super high growth opportunities for AI. I have $40k to invest in ETFs, index funds, and more. Please create a table in markdown.",
    all_cores=True,
)
