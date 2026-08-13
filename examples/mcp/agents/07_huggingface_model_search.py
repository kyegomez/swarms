"""
Example 7 — Hugging Face MCP (free, optional token).

The Hugging Face Hub exposes an MCP server for searching models, datasets,
Spaces, and papers. It works anonymously; adding a free token raises limits
and exposes tools that touch your own account.

    Server : https://huggingface.co/mcp
    Auth   : none required (optional HF token unlocks more)
    Tools  : model_search, dataset_search, space_search, paper_search, ...

This example shows the *optional auth* pattern: the same agent works with or
without a key, and only attaches the Bearer token when one is present. That
matters because a missing environment variable should degrade to anonymous
access, not crash on startup.

Run:
    export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY, etc.
    export HF_TOKEN=...              # optional, free at
                                     # https://huggingface.co/settings/tokens
    python examples/mcp/agents/07_huggingface_model_search.py
"""

import os

from swarms import Agent

MODEL = "gpt-5.4"

HF_TOKEN = os.getenv("HF_TOKEN")

HF_SYSTEM_PROMPT = (
    "You are a machine learning model scout. Help users find the right model "
    "or dataset on the Hugging Face Hub by searching it directly rather than "
    "recalling names from memory. For each candidate you recommend, report "
    "what the search returned: the exact repo id, task, size or parameter "
    "count, and license. Rank recommendations by fitness for the user's "
    "stated constraints — license, hardware budget, and language or domain "
    "coverage — and say explicitly when a popular model is a poor fit for "
    "those constraints. Never invent a repo id; only cite ones the search "
    "actually returned."
)

agent = Agent(
    agent_name="HuggingFace-Scout",
    agent_description=(
        "Finds models and datasets on the Hugging Face Hub via MCP."
    ),
    system_prompt=HF_SYSTEM_PROMPT,
    model_name=MODEL,
    mcp_url="https://huggingface.co/mcp",
    # Optional: anonymous access works, a token just raises the ceiling.
    mcp_api_key=("env:HF_TOKEN" if HF_TOKEN else None),
    max_loops=2,
)

if __name__ == "__main__":
    if not HF_TOKEN:
        print("No HF_TOKEN set - running anonymously (lower rate limits).\n")

    result = agent.run(
        "Find three open-weight embedding models under 500M parameters that "
        "are permissively licensed for commercial use. For each, give the "
        "repo id, parameter count, and license."
    )
    print(result)
