"""
GroupChat of Kimi K3 agents.

Three agents with distinct personas (optimist, skeptic, realist) discuss a
question and reach a conclusion. This is the "multi-agent GroupChat" step from
the "Getting Started with Kimi K3 in Swarms" tutorial.

Each agent listens to the others and decides on its own whether to chime in.
``GroupChat`` handles speaker selection, message passing, and the stopping
condition — ``auto_equip`` (on by default) injects the RESPOND_TOOL into every
agent, so there is no extra tooling to attach.

Set your OpenRouter API key before running:
    export OPENROUTER_API_KEY="your-api-key"

Run:
    uv run examples/multi_agent/groupchat/kimi_k3_groupchat.py
"""

from dotenv import load_dotenv

from swarms import Agent
from swarms.structs.groupchat import GroupChat

load_dotenv()

MODEL = "openrouter/moonshotai/kimi-k3"

optimist = Agent(
    agent_name="Optimist",
    system_prompt="You argue for the opportunities and upside.",
    model_name=MODEL,
    max_loops=1,
    persistent_memory=False,
)

skeptic = Agent(
    agent_name="Skeptic",
    system_prompt="You stress-test claims and surface the risks.",
    model_name=MODEL,
    max_loops=1,
    persistent_memory=False,
)

realist = Agent(
    agent_name="Realist",
    system_prompt="You weigh both sides and push for a decision.",
    model_name=MODEL,
    max_loops=1,
    persistent_memory=False,
)

chat = GroupChat(
    name="Kimi Roundtable",
    agents=[optimist, skeptic, realist],
    max_loops=6,
)

result = chat.run(
    "Should an early-stage startup build its own agents or buy a platform?"
)

print(result)
