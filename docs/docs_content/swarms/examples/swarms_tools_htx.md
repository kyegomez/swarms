# Swarms Tools Example with HTX + CoinGecko

- `pip3 install swarms swarms-tools`
- Add `OPENAI_API_KEY` to your `.env` file

```python
from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)
from swarms_tools import (
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
    return_step_meta=False,
    output_type="str",  # "json", "dict", "csv" OR "string" "yaml" and
    auto_generate_prompt=False,  # Auto generate prompt for the agent based on name, description, and system prompt, task
    max_tokens=4000,  # max output tokens
    saved_state_path="agent_00.json",
    interactive=False,
)

agent.run(
    f"Analyze the $swarms token on HTX with data: {fetch_htx_data('swarms')}. Additionally, consider the following CoinGecko data: {coin_gecko_coin_api('swarms')}"
)
```