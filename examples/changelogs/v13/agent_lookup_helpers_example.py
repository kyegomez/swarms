"""Agent lookup helpers and SerializableMixin.

- find_agent_by_id: shared id-based lookup used across registry, base
  swarm, and utils.
- find_agent_by_name: now backed by a cached name index instead of
  repeated linear scans.
- return_all_agent_names: renamed from get_all_agent_names.
- to_dict(): available on any structure inheriting SerializableMixin
  (GroupChat, HeavySwarm, AgentRearrange, SwarmRouter, ...).
"""

from swarms import Agent, GroupChat
from swarms.structs.ma_blocks import (
    find_agent_by_id,
    find_agent_by_name,
    return_all_agent_names,
)

agents = [
    Agent(agent_name="Analyst", model_name="gpt-5.4", max_loops=1),
    Agent(agent_name="Writer", model_name="gpt-5.4", max_loops=1),
]

# Cached name index — repeated lookups are O(1) after the first call
analyst = find_agent_by_name(agents, agent_name="Analyst")
print(f"Found by name: {analyst.agent_name}")

# Shared id-based lookup
same_agent = find_agent_by_id(agents, agent_id=analyst.id)
print(f"Found by id: {same_agent.agent_name}")

# Renamed from get_all_agent_names
print(f"All names: {return_all_agent_names(agents)}")

# Any SerializableMixin structure serialises with to_dict()
chat = GroupChat(agents=agents, max_loops=5)
state = chat.to_dict()
print(f"Serialised keys: {sorted(state)[:5]} ...")
