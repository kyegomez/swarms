# Swarms Tools Example with HTX + CoinGecko

- `pip3 install swarms swarms-tools`
- Add `OPENAI_API_KEY` to your `.env` file
- Run `swarms_tools_htx_gecko.py`
- Agent will make a function call to the desired tool
- The tool will be executed and the result will be returned to the agent
- The agent will then analyze the result and return the final output


```python
from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)
from swarms_tools import (
    fetch_stock_news,
    coin_gecko_coin_api,
    fetch_htx_data,
)

# Initialize the agent
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    agent_description="Personal finance advisor agent",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    max_loops=1,
    model_name="gpt-4o",
    dynamic_temperature_enabled=True,
    user_name="swarms_corp",
    retry_attempts=3,
    context_length=8192,
    return_step_meta=False,
    output_type="str",  # "json", "dict", "csv" OR "string" "yaml" and
    auto_generate_prompt=False,  # Auto generate prompt for the agent based on name, description, and system prompt, task
    max_tokens=4000,  # max output tokens
    saved_state_path="agent_00.json",
    interactive=False,
    tools=[fetch_stock_news, coin_gecko_coin_api, fetch_htx_data],
)

agent.run("Analyze the $swarms token on htx")
```