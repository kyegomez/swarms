from swarms import Agent

tools = [
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two numbers together and return the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the operation to perform.",
                    },
                    "a": {
                        "type": "integer",
                        "description": "The first number to add.",
                    },
                    "b": {
                        "type": "integer",
                        "description": "The second number to add.",
                    },
                },
                "required": [
                    "name",
                    "a",
                    "b",
                ],
            },
        },
    }
]


# Initialize the agent
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    agent_description="Personal finance advisor agent",
    max_loops=2,
    tools_list_dictionary=tools,
    output_type="final",
    mcp_url="http://0.0.0.0:8000/sse",
)

out = agent.run(
    "Use the multiply tool to multiply 3 and 4 together. Look at the tools available to you.",
)

print(agent.short_memory.get_str())
