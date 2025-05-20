from swarms.structs.aop import AOP

aop = AOP(
    name="example_system",
    description="A simple example of tools, agents, and swarms",
    url="http://localhost:8000/sse",
)

# print(
#     aop.call_tool_or_agent(
#         url="http://localhost:8000/sse",
#         name="calculator",
#         arguments={"operation": "add", "x": 1, "y": 2},
#         output_type="list",
#     )
# )


# print(
#     aop.call_tool_or_agent_batched(
#         url="http://localhost:8000/sse",
#         names=["calculator", "calculator"],
#         arguments=[{"operation": "add", "x": 1, "y": 2}, {"operation": "multiply", "x": 3, "y": 4}],
#         output_type="list",
#     )
# )


# print(
#     aop.call_tool_or_agent_concurrently(
#         url="http://localhost:8000/sse",
#         names=["calculator", "calculator"],
#         arguments=[{"operation": "add", "x": 1, "y": 2}, {"operation": "multiply", "x": 3, "y": 4}],
#         output_type="list",
#     )
# )


# print(aop.list_agents())

# print(aop.list_tools())

# print(aop.list_swarms())

# print(aop.list_all(url="http://localhost:8000/sse"))

# print(any_to_str(aop.list_all()))

# print(aop.search_if_tool_exists(name="calculator"))

# out = aop.list_tool_parameters(name="calculator")
# print(type(out))
# print(out)

print(aop.list_agents())
print(aop.list_swarms())
