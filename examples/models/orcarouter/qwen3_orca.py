from dotenv import load_dotenv

from swarms import Agent

load_dotenv()

# Initialize a Qwen3 agent routed through OrcaRouter.
# OrcaRouter exposes an OpenAI-compatible endpoint, so model names can be
# prefixed with ``orcarouter/`` and are resolved via the ORCAROUTER_API_KEY
# environment variable — no base URL wiring required.
agent = Agent(
    agent_name="Qwen3-OrcaRouter-Agent",
    agent_description="Qwen3 VL model routed through OrcaRouter",
    system_prompt="You are a helpful assistant that answers questions concisely.",
    model_name="orcarouter/qwen/qwen3-vl-235b-a22b-instruct",
    max_loops=1,
)

out = agent.run(
    task="Explain the difference between a swarm and a single agent in two sentences.",
)

print(out)
