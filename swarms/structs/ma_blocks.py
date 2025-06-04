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
