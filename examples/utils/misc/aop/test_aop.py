from swarms.structs.aop import AOP

# Initialize the AOP instance
aop = AOP(
    name="example_system",
    description="A simple example of tools, agents, and swarms",
)


# Define a simple tool
@aop.tool(name="calculator", description="A simple calculator tool")
async def calculator(operation: str, x: float, y: float):
    """
    Performs basic arithmetic operations
    """
    if operation == "add":
        return x + y
    elif operation == "multiply":
        return x * y
    else:
        raise ValueError("Unsupported operation")


# Define an agent that uses the calculator tool
@aop.agent(
    name="math_agent",
    description="Agent that performs mathematical operations",
)
async def math_agent(operation: str, numbers: list[float]):
    """
    Agent that chains multiple calculations together
    """
    result = numbers[0]
    for num in numbers[1:]:
        # Using the calculator tool within the agent
        result = await aop.call_tool_or_agent(
            "calculator",
            {"operation": operation, "x": result, "y": num},
        )
    return result


# Define a swarm that coordinates multiple agents
@aop.swarm(
    name="math_swarm",
    description="Swarm that coordinates mathematical operations",
)
async def math_swarm(numbers: list[float]):
    """
    Swarm that performs multiple operations on a set of numbers
    """
    # Perform addition and multiplication in parallel
    results = await aop.call_tool_or_agent_concurrently(
        names=["math_agent", "math_agent"],
        arguments=[
            {"operation": "add", "numbers": numbers},
            {"operation": "multiply", "numbers": numbers},
        ],
    )

    return {"sum": results[0], "product": results[1]}


# Example usage
if __name__ == "__main__":
    aop.run_sse()
