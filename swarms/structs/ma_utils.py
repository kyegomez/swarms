from typing import Dict, List, Any, Optional, Union, Callable
import random
from swarms.prompts.collaborative_prompts import (
    get_multi_agent_collaboration_prompt_one,
)
from functools import lru_cache

from loguru import logger


def list_all_agents(
    agents: List[Union[Callable, Any]],
    conversation: Optional[Any] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    add_to_conversation: Optional[bool] = False,
    add_collaboration_prompt: Optional[bool] = True,
) -> str:
    """Lists all agents in a swarm and optionally adds them to a conversation.

    This function compiles information about all agents in a swarm, including their names and descriptions.
    It can optionally add this information to a conversation history.

    Args:
        agents (List[Union[Agent, Any]]): List of agents to list information about
        conversation (Any): Conversation object to optionally add agent information to
        name (str): Name of the swarm/group of agents
        add_to_conversation (bool, optional): Whether to add agent information to conversation. Defaults to False.

    Returns:
        str: Formatted string containing information about all agents

    Example:
        >>> agents = [agent1, agent2]
        >>> conversation = Conversation()
        >>> agent_info = list_all_agents(agents, conversation, "MySwarm")
        >>> print(agent_info)
        Swarm: MySwarm
        Total Agents: 2

        Agent: Agent1
        Description: First agent description...

        Agent: Agent2
        Description: Second agent description...
    """

    # Compile information about all agents
    total_agents = len(agents)

    all_agents = f"Team Name: {name}\n" if name else ""
    all_agents += (
        f"Team Description: {description}\n" if description else ""
    )
    all_agents += f"Total Agents: {total_agents}\n\n"
    all_agents += "| Agent | Description |\n"
    all_agents += "|-------|-------------|\n"
    all_agents += "\n".join(
        f"| {agent.agent_name} | {agent.description or (agent.system_prompt[:50] + '...' if len(agent.system_prompt) > 50 else agent.system_prompt)} |"
        for agent in agents
    )

    if add_to_conversation:
        # Add the agent information to the conversation
        conversation.add(
            role="System",
            content=all_agents,
        )

    if add_collaboration_prompt:
        return get_multi_agent_collaboration_prompt_one(
            agents_in_swarm=all_agents
        )
    else:
        return all_agents


models = [
    "anthropic/claude-3-sonnet-20240229",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "deepseek/deepseek-chat",
    "deepseek/deepseek-reasoner",
    "groq/deepseek-r1-distill-qwen-32b",
    "groq/deepseek-r1-distill-qwen-32b",
    # "gemini/gemini-pro",
    # "gemini/gemini-1.5-pro",
    "openai/03-mini",
    "o4-mini",
    "o3",
    "gpt-4.1",
    "groq/llama-3.1-8b-instant",
    "gpt-4.1-nano",
]


def set_random_models_for_agents(
    agents: Optional[Union[List[Callable], Callable]] = None,
    model_names: List[str] = models,
) -> Union[List[Callable], Callable, str]:
    """Sets random models for agents in the swarm or returns a random model name.

    Args:
        agents (Optional[Union[List[Agent], Agent]]): Either a single agent, list of agents, or None
        model_names (List[str], optional): List of model names to choose from. Defaults to models.

    Returns:
        Union[List[Agent], Agent, str]: The agent(s) with randomly assigned models or a random model name
    """
    if agents is None:
        return random.choice(model_names)

    if isinstance(agents, list):
        return [
            setattr(agent, "model_name", random.choice(model_names))
            or agent
            for agent in agents
        ]
    else:
        setattr(agents, "model_name", random.choice(model_names))
        return agents


@lru_cache(maxsize=128)
def _create_agent_map_cached(
    agent_tuple: tuple,
) -> Dict[str, Union[Callable, Any]]:
    """Internal cached version of create_agent_map that takes a tuple for hashability."""
    try:
        return {
            (
                agent.agent_name
                if isinstance(agent, Callable)
                else agent.__name__
            ): agent
            for agent in agent_tuple
        }
    except (AttributeError, TypeError) as e:
        logger.error(f"Error creating agent map: {e}")
        return {}


def create_agent_map(
    agents: List[Union[Callable, Any]],
) -> Dict[str, Union[Callable, Any]]:
    """Creates a map of agent names to agents for fast lookup.

    This function is optimized with LRU caching to avoid recreating maps for identical agent lists.
    The cache stores up to 128 different agent map configurations.

    Args:
        agents (List[Union[Callable, Any]]): List of agents to create a map of. Each agent should either be:
            - A callable with a __name__ attribute
            - An object with an agent_name attribute

    Returns:
        Dict[str, Union[Callable, Any]]: Map of agent names to agents

    Examples:
        >>> def agent1(): pass
        >>> def agent2(): pass
        >>> agents = [agent1, agent2]
        >>> agent_map = create_agent_map(agents)
        >>> print(agent_map.keys())
        dict_keys(['agent1', 'agent2'])

        >>> class Agent:
        ...     def __init__(self, name):
        ...         self.agent_name = name
        >>> agents = [Agent("bot1"), Agent("bot2")]
        >>> agent_map = create_agent_map(agents)
        >>> print(agent_map.keys())
        dict_keys(['bot1', 'bot2'])

    Raises:
        ValueError: If agents list is empty
        TypeError: If any agent lacks required name attributes
    """
    if not agents:
        raise ValueError("Agents list cannot be empty")

    # Convert list to tuple for hashability
    return _create_agent_map_cached(tuple(agents))
