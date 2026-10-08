"""
Single-agent example using OpenRouter Moonshotai Kimi K3.

A concise research/analysis assistant — the "single agent" step from the
"Getting Started with Kimi K3 in Swarms" tutorial.

Set your OpenRouter API key before running:
    export OPENROUTER_API_KEY="your-api-key"

Run:
    uv run examples/models/openrouter/kimi_k3_single_agent.py
"""

from dotenv import load_dotenv

from swarms import Agent

load_dotenv()

agent = Agent(
    agent_name="Research-Agent",
    agent_description="A concise research and analysis assistant",
    system_prompt=(
        "You are a sharp research analyst. Answer clearly, cite the key "
        "trade-offs, and state any assumptions you make."
    ),
    model_name="openrouter/moonshotai/kimi-k3",
    max_loops=1,
    reasoning_effort="low",
)

result = agent.run(
    task="Explain the trade-offs between vector databases and keyword search for RAG."
)

print(result)
