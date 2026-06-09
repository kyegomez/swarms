from typing import Union
from swarms.structs.agent import Agent
from typing import List, Callable
from swarms.structs.conversation import Conversation
from swarms.structs.multi_agent_exec import run_agents_concurrently
from swarms.utils.history_output_formatter import (
    history_output_formatter,
    HistoryOutputType,
)

from swarms.prompts.agent_conversation_aggregator import (
    AGGREGATOR_SYSTEM_PROMPT,
)


def aggregator_agent_task_prompt(
    task: str, workers: List[Agent], conversation: Conversation
):
    return f"""
    Please analyze and summarize the following multi-agent conversation, following your guidelines for comprehensive synthesis:

    Conversation Context:
    Original Task: {task}
    Number of Participating Agents: {len(workers)}

    Conversation Content:
    {conversation.get_str()}

    Please provide a 3,000 word comprehensive summary report of the conversation.
    """


def aggregate(
    workers: List[Callable],
    task: str = None,
    type: HistoryOutputType = "all",
    aggregator_model_name: str = "anthropic/claude-3-sonnet-20240229",
):
    """
    Aggregate a list of tasks into a single task.
    """

    if task is None:
        raise ValueError("Task is required in the aggregator block")

    if workers is None:
        raise ValueError(
            "Workers is required in the aggregator block"
        )

    if not isinstance(workers, list):
        raise ValueError("Workers must be a list of Callable")

    if not all(isinstance(worker, Callable) for worker in workers):
        raise ValueError("Workers must be a list of Callable")

    conversation = Conversation()

    aggregator_agent = Agent(
        agent_name="Aggregator",
        agent_description="Expert agent specializing in analyzing and synthesizing multi-agent conversations",
        system_prompt=AGGREGATOR_SYSTEM_PROMPT,
        max_loops=1,
        model_name=aggregator_model_name,
        output_type="final",
        max_tokens=4000,
    )

    results = run_agents_concurrently(agents=workers, task=task)

    # Zip the results with the agents
    for result, agent in zip(results, workers):
        conversation.add(content=result, role=agent.agent_name)

    final_result = aggregator_agent.run(
        task=aggregator_agent_task_prompt(task, workers, conversation)
    )

    conversation.add(
        content=final_result, role=aggregator_agent.agent_name
    )

    return history_output_formatter(
        conversation=conversation, type=type
    )


def run_agent(
    agent: Agent,
    task: str,
    type: HistoryOutputType = "all",
    *args,
    **kwargs,
):
    """
    Run an agent on a task.

    Args:
        agent (Agent): The agent to run
        task (str): The task to run the agent on
        type (HistoryOutputType, optional): The type of history output. Defaults to "all".
        *args: Variable length argument list
        **kwargs: Arbitrary keyword arguments

    Returns:
        Any: The result of running the agent

    Raises:
        ValueError: If agent or task is None
        TypeError: If agent is not an instance of Agent
    """
    if agent is None:
        raise ValueError("Agent cannot be None")

    if task is None:
        raise ValueError("Task cannot be None")

    if not isinstance(agent, Agent):
        raise TypeError("Agent must be an instance of Agent")

    try:
        return agent.run(task=task, *args, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Error running agent: {str(e)}")


_FIND_AGENT_INDEX_CACHE: dict = {}
_FIND_AGENT_INDEX_CACHE_MAX = 256


def _build_agent_name_index(agents):
    index = {}
    for agent in agents:
        name = getattr(agent, "agent_name", None)
        if name is not None:
            index[name] = agent
        alt = getattr(agent, "name", None)
        if alt is not None and alt != name:
            index[alt] = agent
    return index


def find_agent_by_name(
    agents: List[Union[Agent, Callable]], agent_name: str
) -> Agent:
    """
    Find an agent by its name in a list of agents.

    Builds a name -> agent index on first call for a given list and
    reuses it on subsequent calls, turning repeated lookups from O(n)
    into O(1).

    Args:
        agents (List[Union[Agent, Callable]]): List of agents to search through
        agent_name (str): Name of the agent to find

    Returns:
        Agent: The found agent

    Raises:
        ValueError: If agents list is empty or agent not found
        TypeError: If agent_name is not a string
    """
    if not agents:
        raise ValueError("Agents list cannot be empty")

    if not isinstance(agent_name, str):
        raise TypeError("Agent name must be a string")

    if not agent_name.strip():
        raise ValueError("Agent name cannot be empty or whitespace")

    key = id(agents)
    length = len(agents)
    cached = _FIND_AGENT_INDEX_CACHE.get(key)
    if cached is None or cached[0] != length:
        if (
            len(_FIND_AGENT_INDEX_CACHE)
            >= _FIND_AGENT_INDEX_CACHE_MAX
        ):
            _FIND_AGENT_INDEX_CACHE.clear()
        index = _build_agent_name_index(agents)
        _FIND_AGENT_INDEX_CACHE[key] = (length, index)
    else:
        index = cached[1]

    agent = index.get(agent_name)
    if agent is not None:
        return agent

    index = _build_agent_name_index(agents)
    _FIND_AGENT_INDEX_CACHE[key] = (length, index)
    agent = index.get(agent_name)
    if agent is None:
        raise ValueError(f"Agent with name '{agent_name}' not found")
    return agent


def find_agent_by_id(
    agents: List[Union["Agent", Callable]],
    agent_id: str,
) -> Agent:
    """
    Find an agent by its id in a list of agents.

    Args:
        agents (List[Union[Agent, Callable]]): The list of agent objects to search through.
        agent_id (str): The unique identifier of the agent to find.

    Returns:
        Agent: The agent object with the matching id, or None if no match is found.
    """
    return next(
        (agent for agent in agents if agent.id == agent_id), None
    )


def find_multiple_agents_by_name(
    agents: List[Union["Agent", Callable]],
    agent_names: List[str],
) -> List[Agent]:
    """
    Find multiple agents by their names in a list of agents.

    Args:
        agents (List[Union[Agent, Callable]]): The list of agent objects to search through.
        agent_names (List[str]): A list containing the names of agents to find.

    Returns:
        List[Agent]: A list of agent objects whose names are in agent_names.
    """
    return [
        agent for agent in agents if agent.agent_name in agent_names
    ]


def return_all_agent_names(
    agents: List[Union["Agent", Callable]],
) -> List[str]:
    """
    Return all agent names from a list of agents.
    Uses map for speed (avoids interpreter loop overhead of list comp).
    """
    attr = getattr  # local reference for faster attribute access
    return list(map(lambda a: attr(a, "agent_name"), agents))
