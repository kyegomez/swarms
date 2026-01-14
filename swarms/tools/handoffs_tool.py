"""
Handoffs tool for agent-to-agent task delegation.

This module provides a tool function that allows agents to dynamically
delegate tasks to other specialized agents.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from loguru import logger


def handoff_task(
    handoffs: List[Dict[str, str]],
    agent_registry: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Delegate tasks to one or more specialized agents.

    This tool allows the current agent to dynamically send tasks to other agents
    based on their capabilities. The agent can delegate to a single agent or
    multiple agents simultaneously for parallel processing.

    Args:
        handoffs: List of handoff requests. Each request is a dictionary containing:
            - agent_name (str): The name of the agent to delegate to
            - task (str): The task to be delegated
            - reasoning (str): Explanation of why this agent was selected
        agent_registry: Dictionary mapping agent names to agent instances.
                       This is automatically provided by the Agent class and should
                       not be passed manually.

    Returns:
        str: A formatted string containing responses from all delegated agents.

    Example:
        >>> handoff_task(
        ...     handoffs=[
        ...         {
        ...             "agent_name": "ResearchAgent",
        ...             "task": "Research the latest AI developments",
        ...             "reasoning": "ResearchAgent specializes in research tasks"
        ...         },
        ...         {
        ...             "agent_name": "CodeExpertAgent",
        ...             "task": "Write a Python function for data processing",
        ...             "reasoning": "CodeExpertAgent specializes in coding tasks"
        ...         }
        ...     ],
        ...     agent_registry={"ResearchAgent": research_agent, "CodeExpertAgent": code_agent}
        ... )
    """
    if not agent_registry:
        return "Error: No agent registry provided. Handoffs are not configured."

    if not handoffs:
        return "Error: No handoffs specified."

    if not isinstance(handoffs, list):
        return "Error: handoffs must be a list of handoff requests."

    results = []
    errors = []

    # Validate all agent names before execution
    for handoff in handoffs:
        if not isinstance(handoff, dict):
            errors.append(
                "Each handoff must be a dictionary with agent_name, task, and reasoning"
            )
            continue

        agent_name = handoff.get("agent_name")
        task = handoff.get("task")
        reasoning = handoff.get("reasoning")

        if not agent_name:
            errors.append("One or more handoffs missing 'agent_name'")
            continue

        if not task:
            errors.append(f"Handoff to {agent_name} missing 'task'")
            continue

        if agent_name not in agent_registry:
            available_agents = list(agent_registry.keys())
            errors.append(
                f"Agent '{agent_name}' not found in available agents. "
                f"Available agents: {available_agents}"
            )
            continue

    if errors:
        return "Validation errors:\n" + "\n".join(
            f"- {error}" for error in errors
        )

    # Execute handoffs
    if len(handoffs) == 1:
        # Single agent - execute directly
        handoff = handoffs[0]
        agent_name = handoff["agent_name"]
        task = handoff.get("task", "")
        reasoning = handoff.get("reasoning", "")

        try:
            agent = agent_registry[agent_name]
            logger.info(
                f"Delegating task to {agent_name}: {task[:100]}..."
            )

            # Execute the task
            response = agent.run(task=task)

            result = f"""
            Agent: {agent_name}
            
            Reasoning: {reasoning}
            
            Task: {task}
            
            Response:
            {response}
            """
            results.append(result)

        except Exception as e:
            error_msg = (
                f"Error executing handoff to {agent_name}: {str(e)}"
            )
            logger.error(error_msg)
            errors.append(error_msg)

    else:
        # Multiple agents - execute in parallel
        with ThreadPoolExecutor(
            max_workers=len(handoffs)
        ) as executor:
            future_to_handoff = {}

            for handoff in handoffs:
                agent_name = handoff["agent_name"]
                task = handoff.get("task", "")

                if agent_name in agent_registry:
                    agent = agent_registry[agent_name]
                    future = executor.submit(agent.run, task=task)
                    future_to_handoff[future] = handoff

            # Collect results as they complete
            for future in as_completed(future_to_handoff):
                handoff = future_to_handoff[future]
                agent_name = handoff["agent_name"]
                task = handoff.get("task", "")
                reasoning = handoff.get("reasoning", "")

                try:
                    response = future.result()
                    result = f"""
Agent: {agent_name}
Reasoning: {reasoning}
Task: {task}
Response:
{response}
"""
                    results.append(result)
                    logger.info(f"Completed handoff to {agent_name}")

                except Exception as e:
                    error_msg = f"Error executing handoff to {agent_name}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)

    # Format final response
    if results:
        response_text = "=" * 80 + "\n"
        response_text += "HANDOFF RESULTS\n"
        response_text += "=" * 80 + "\n\n"
        response_text += "\n".join(results)

        if errors:
            response_text += "\n\n" + "=" * 80 + "\n"
            response_text += "ERRORS\n"
            response_text += "=" * 80 + "\n"
            response_text += "\n".join(
                f"- {error}" for error in errors
            )

        return response_text
    else:
        return "No successful handoffs. Errors:\n" + "\n".join(
            f"- {error}" for error in errors
        )
