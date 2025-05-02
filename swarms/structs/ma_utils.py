from swarms.structs.agent import Agent
from typing import List, Any, Optional, Union
import random


def list_all_agents(
    agents: List[Union[Agent, Any]],
    conversation: Optional[Any] = None,
    name: str = "",
    add_to_conversation: bool = False,
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
        Total Agents: 2

        Agent: Agent1
        Description: First agent description...

        Agent: Agent2
        Description: Second agent description...
    """

    # Compile information about all agents
    total_agents = len(agents)

    all_agents = f"Total Agents: {total_agents}\n\n" + "\n\n".join(
        f"Agent: {agent.agent_name} \n\n Description: {agent.description or (agent.system_prompt[:50] + '...' if len(agent.system_prompt) > 50 else agent.system_prompt)}"
        for agent in agents
    )

    if add_to_conversation:
        # Add the agent information to the conversation
        conversation.add(
            role="System",
            content=f"All Agents Available in the Swarm {name}:\n\n{all_agents}",
        )

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
    "gpt-4.1-nano",
]


def set_random_models_for_agents(
    agents: Union[List[Agent], Agent], model_names: List[str] = models
) -> Union[List[Agent], Agent]:
    """Sets random models for agents in the swarm.

    Args:
        agents (Union[List[Agent], Agent]): Either a single agent or a list of agents
        model_names (List[str], optional): List of model names to choose from. Defaults to models.

    Returns:
        Union[List[Agent], Agent]: The agent(s) with randomly assigned models
    """
    if isinstance(agents, list):
        return [
            setattr(agent, "model_name", random.choice(model_names))
            or agent
            for agent in agents
        ]
    else:
        setattr(agents, "model_name", random.choice(model_names))
        return agents
