from typing import List, Any
from loguru import logger
from swarms.structs.agent import Agent


def get_agent_name(agent: Any) -> str:
    """Helper function to safely get agent name

    Args:
        agent (Any): The agent object to get name from

    Returns:
        str: The agent's name if found, 'Unknown' otherwise
    """
    if isinstance(agent, Agent) and hasattr(agent, "agent_name"):
        return agent.agent_name
    return "Unknown"


def get_agent_description(agent: Any) -> str:
    """Helper function to get agent description or system prompt preview

    Args:
        agent (Any): The agent object

    Returns:
        str: Description or first 100 chars of system prompt
    """
    if not isinstance(agent, Agent):
        return "N/A"

    if hasattr(agent, "description") and agent.description:
        return agent.description

    if hasattr(agent, "system_prompt") and agent.system_prompt:
        return f"{agent.system_prompt[:150]}..."

    return "N/A"


def showcase_available_agents(
    name: str = None,
    description: str = None,
    agents: List[Agent] = [],
    update_agents_on: bool = False,
) -> str:
    """
    Generate a formatted string showcasing all available agents and their descriptions.

    Args:
        agents (List[Agent]): List of Agent objects to showcase.
        update_agents_on (bool, optional): If True, updates each agent's system prompt with
            the showcase information. Defaults to False.

    Returns:
        str: Formatted string containing agent information, including names, descriptions
            and IDs for all available agents.
    """
    logger.info(f"Showcasing {len(agents)} available agents")

    formatted_agents = []
    header = f"\n####### Agents available in the swarm: {name} ############\n"
    header += f"{description}\n"
    row_format = "{:<5} | {:<20} | {:<50}"
    header_row = row_format.format("ID", "Agent Name", "Description")
    separator = "-" * 80

    formatted_agents.append(header)
    formatted_agents.append(separator)
    formatted_agents.append(header_row)
    formatted_agents.append(separator)

    for idx, agent in enumerate(agents):
        if not isinstance(agent, Agent):
            logger.warning(
                f"Skipping non-Agent object: {type(agent)}"
            )
            continue

        agent_name = get_agent_name(agent)
        description = (
            get_agent_description(agent)[:100] + "..."
            if len(get_agent_description(agent)) > 100
            else get_agent_description(agent)
        )

        formatted_agents.append(
            row_format.format(idx + 1, agent_name, description)
        )

    showcase = "\n".join(formatted_agents)

    return showcase
