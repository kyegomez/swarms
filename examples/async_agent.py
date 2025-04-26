from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)

# Initialize the agent (no swarm_models import needed)
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    agent_description="Personal finance advisor agent",
    system_prompt=(
        FINANCIAL_AGENT_SYS_PROMPT
        + " Output the <DONE> token when you're done creating a portfolio"
    ),
    max_loops=1,
    model_name="gpt-4o",
    dynamic_temperature_enabled=True,
    user_name="Kye",
    retry_attempts=3,
    # streaming_on=True,
    context_length=8192,
    return_step_meta=False,
    output_type="str",  # "json", "dict", "csv" OR "string" "yaml" and
    auto_generate_prompt=False,  # Auto generate prompt for the agent based on name, description, and system prompt, task
    max_tokens=4000,  # max output tokens
    # interactive=True,
    stopping_token="<DONE>",
    saved_state_path="agent_00.json",
    interactive=False,
)


async def run_agent():
    await agent.arun(
        "Create a table of super high-growth AI investment opportunities "
        "with $40k in ETFs, index funds, etc., and output it in Markdown.",
        all_cores=True,
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_agent())
