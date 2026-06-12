"""Agent.tools_list_dictionary defaults to an empty list.

Previously None — callers needed a None check before appending tool
schemas. Now it is always a list.
"""

from swarms import Agent
from swarms.structs.groupchat import RESPOND_TOOL

agent = Agent(agent_name="A", model_name="gpt-5.4", max_loops=1)

# [] — append schemas without a None check
print(f"Default: {agent.tools_list_dictionary}")

agent.tools_list_dictionary.append(RESPOND_TOOL)
print(
    f"After append: {len(agent.tools_list_dictionary)} tool schema(s)"
)
