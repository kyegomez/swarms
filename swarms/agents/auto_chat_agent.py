from swarms.structs.agent import Agent
from typing import Optional


def auto_chat_agent(
    name: str = "Swarms Agent",
    description: str = "A Swarms agent that can chat with the user.",
    system_prompt: Optional[str] = None,
    task: Optional[str] = None,
) -> Agent:
    """
    Create an auto chat agent.
    """
    agent = Agent(
        agent_name=name,
        agent_description=description,
        system_prompt=system_prompt,
        interactive=True,
        dynamic_context_window=True,
        dynamic_temperature_enabled=True,
        context_length=100000,
        max_loops="auto",
    )

    # Pass task explicitly - if None and interactive=True, will prompt for input
    return agent.run(task=task)
