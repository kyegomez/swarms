"""
Handoffs prompt for agent-to-agent task delegation.

This module provides prompts and instructions for agents to dynamically
delegate tasks to other specialized agents.
"""


def get_handoffs_prompt(available_agents: list) -> str:
    """
    Generate a system prompt for handoffs functionality.

    Args:
        available_agents: List of available agents with their names and descriptions.
                         Each agent should have 'agent_name' and 'agent_description' attributes.

    Returns:
        str: System prompt for handoffs functionality.
    """
    # Build agent descriptions list
    agent_descriptions = []
    for agent in available_agents:
        name = getattr(agent, "agent_name", str(agent))
        description = getattr(
            agent, "agent_description", "No description available"
        )
        agent_descriptions.append(f"- {name}: {description}")

    agents_list = (
        "\n".join(agent_descriptions)
        if agent_descriptions
        else "No agents available"
    )

    return f"""
You have the ability to delegate tasks to specialized agents when appropriate. This allows you to leverage the expertise of other agents to complete complex or multi-faceted tasks more effectively.

**Available Agents:**
{agents_list}

**When to Use Handoffs:**
1. **Task Specialization**: When a task requires specific expertise that another agent possesses
2. **Task Complexity**: When a task can be broken down into subtasks that different agents can handle
3. **Parallel Processing**: When multiple independent tasks can be executed simultaneously by different agents
4. **Resource Optimization**: When delegating would be more efficient than handling the task yourself

**How to Use Handoffs:**
- Use the `handoff_task` tool to delegate tasks to one or more agents
- You can send the same task to multiple agents for different perspectives
- You can send different tasks to different agents for parallel execution
- Always provide clear reasoning for why you're delegating and what you expect from the agent(s)

**Best Practices:**
- Analyze the task requirements before deciding to delegate
- Choose agents whose capabilities match the task requirements
- Provide clear, actionable task descriptions to the receiving agents
- Consider whether the task needs to be modified for the receiving agent
- Use multiple agents when the task has distinct components requiring different expertise

**Important Notes:**
- You can delegate to one or more agents simultaneously
- Each agent will execute their assigned task independently
- You will receive responses from all agents that were delegated to
- Use the agent names exactly as listed above
- If you're unsure whether to delegate, consider the complexity and your own capabilities first

Remember: Handoffs are a powerful tool for collaboration, but use them judiciously. Only delegate when it truly improves task completion quality or efficiency.
"""


def get_handoffs_instruction_prompt() -> str:
    """
    Get a concise instruction prompt for handoffs.

    Returns:
        str: Brief instruction prompt for handoffs.
    """
    return """
You can delegate tasks to specialized agents using the handoff_task tool. 
When a task requires specific expertise or can be better handled by another agent, 
use the handoff_task tool to send the task to the appropriate agent(s).
"""
