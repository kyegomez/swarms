"""Zena: CSV agents folded into the agent loader.

The standalone CSV-to-agent module is gone. Loading from CSV is part of
AgentLoader now, alongside the Markdown loader — one import, one API, one
set of validation rules.
"""

from swarms.structs.agent_loader import AgentLoader

loader = AgentLoader()

agents = loader.load_agents_from_csv("agents.csv")

for agent in agents:
    print(agent.agent_name, "->", agent.model_name)
