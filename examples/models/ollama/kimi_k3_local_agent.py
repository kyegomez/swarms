"""
Run Kimi K3 locally with Ollama.

The same single-agent example as the OpenRouter version, but served by a local
Ollama model — the only change is the ``model_name`` string. This is the "run
locally with Ollama" step from the "Getting Started with Kimi K3 in Swarms"
tutorial.

Pull and serve the model first:
    ollama run kimi-k3:cloud

Then run:
    uv run examples/models/ollama/kimi_k3_local_agent.py
"""

from swarms import Agent

agent = Agent(
    agent_name="Research-Agent",
    agent_description="A concise research and analysis assistant",
    system_prompt=(
        "You are a sharp research analyst. Answer clearly, cite the key "
        "trade-offs, and state any assumptions you make."
    ),
    model_name="ollama/kimi-k3:cloud",  # local via Ollama; no API key needed
    max_loops=1,
    reasoning_effort="low",
)

result = agent.run(
    task="Explain the trade-offs between vector databases and keyword search for RAG."
)

print(result)
