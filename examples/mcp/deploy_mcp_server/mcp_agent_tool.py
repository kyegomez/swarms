from mcp.server.fastmcp import FastMCP
from swarms import Agent

mcp = FastMCP("MCPAgentTool")

@mcp.tool(
    name="create_agent",
    description="Create an agent with the specified name, system prompt, and model, then run a task.",
)
def create_agent(agent_name: str, system_prompt: str, model_name: str, task: str) -> str:
    """
    Create an agent with the given parameters and execute the specified task.

    Args:
        agent_name (str): The name of the agent to create.
        system_prompt (str): The system prompt to initialize the agent with.
        model_name (str): The model name to use for the agent.
        task (str): The task for the agent to perform.

    Returns:
        str: The result of the agent running the given task.
    """
    agent = Agent(
        agent_name=agent_name,
        system_prompt=system_prompt,
        model_name=model_name,
    )
    return agent.run(task)

if __name__ == "__main__":
    mcp.run(transport="streamable-http")