import json
from pathlib import Path

from dotenv import load_dotenv

from swarms import Agent, AutoAgentBuilder

load_dotenv()

TASK = "Audit a Python codebase for security vulnerabilities and write up the findings."
ROSTER_FILE = Path("roster.json")


def design_roster() -> list[dict]:
    """Call the builder once and cache the result to disk."""
    configs = AutoAgentBuilder(num_agents=3, return_dict=True).run(
        TASK
    )
    ROSTER_FILE.write_text(json.dumps(configs, indent=2))
    print(f"Designed {len(configs)} agents -> {ROSTER_FILE}")
    return configs


def load_roster() -> list[dict]:
    """Read the cached roster. No model call, no cost, same team every time."""
    configs = json.loads(ROSTER_FILE.read_text())
    print(f"Loaded {len(configs)} agents from {ROSTER_FILE}")
    return configs


configs = load_roster() if ROSTER_FILE.exists() else design_roster()

# The configs are plain dicts, so you can edit them before building — pin every
# agent to one model, tighten a system prompt, drop an agent you disagree with.
for config in configs:
    config["model_name"] = "gpt-5.4-mini"

agents = [
    Agent(
        agent_name=config["name"],
        agent_description=config["description"],
        system_prompt=config["system_prompt"],
        model_name=config["model_name"],
        max_loops=1,
    )
    for config in configs
]

print("\nReady to run:")
for agent in agents:
    print(f"  {agent.agent_name}  [{agent.model_name}]")
