from swarms import Agent

from swarms_tools.finance.dex_screener import (
    fetch_dex_screener_profiles,
)

# Initialize the agent
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    agent_description="Personal finance advisor agent",
    max_loops=1,
    model_name="gpt-4o",
    dynamic_temperature_enabled=True,
    user_name="swarms_corp",
    return_step_meta=False,
    output_type="str",  # "json", "dict", "csv" OR "string" "yaml" and
    auto_generate_prompt=False,  # Auto generate prompt for the agent based on name, description, and system prompt, task
    interactive=False,
)

token_profiles = fetch_dex_screener_profiles()
prompt = f"Using data from DexScreener, analyze the latest tokens and provide a detailed analysis with top 5 tokens based on their potential, considering both their profiles and recent boosts. The token profiles are sourced from DexScreener's token profiles API, while the token boosts are sourced from DexScreener's latest token boosts API. {str(token_profiles)}"
agent.run(prompt)
