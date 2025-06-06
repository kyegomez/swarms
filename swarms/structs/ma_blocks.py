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


def find_agent_by_name(
    agents: List[Union[Agent, Callable]], agent_name: str
) -> Agent:
    """
    Find an agent by its name in a list of agents.

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

    try:
        for agent in agents:
            if hasattr(agent, "name") and agent.name == agent_name:
                return agent
        raise ValueError(f"Agent with name '{agent_name}' not found")
    except Exception as e:
        raise RuntimeError(f"Error finding agent: {str(e)}")
