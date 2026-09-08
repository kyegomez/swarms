"""A MixtureOfAgents swarm of gpt-6-astra agents through the Swarms API.

Three specialists answer the task in parallel and a synthesizer combines
their answers. Every agent runs gpt-6-astra.

Run:
    export SWARMS_API_KEY=...   # https://swarms.world/platform/api-keys
    python examples/models/gpt_astra/swarms_api_mixture_of_agents.py
"""

import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("SWARMS_API_KEY")
BASE_URL = "https://api.swarms.world"

headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}

MODEL = "gpt-6-astra"

payload = {
    "name": "Semiconductor-ETF-Council",
    "description": "Three views on a semiconductor ETF question, then a synthesis.",
    "swarm_type": "MixtureOfAgents",
    "max_loops": 1,
    "agents": [
        {
            "agent_name": "Fundamentals-Analyst",
            "description": "Weighs holdings, concentration and expense ratios.",
            "system_prompt": (
                "You are a fund analyst. Compare ETFs on holdings, "
                "concentration and expense ratio. Be concrete and brief."
            ),
            "model_name": MODEL,
            "max_loops": 1,
            "max_tokens": 4000,
        },
        {
            "agent_name": "Risk-Officer",
            "description": "Names the risks an investor is taking on.",
            "system_prompt": (
                "You are a risk officer. State the main risks of each option "
                "and who should avoid it. Be concrete and brief."
            ),
            "model_name": MODEL,
            "max_loops": 1,
            "max_tokens": 4000,
        },
        {
            "agent_name": "Portfolio-Manager",
            "description": "Recommends an allocation for a long-term investor.",
            "system_prompt": (
                "You are a portfolio manager for long-term retail investors. "
                "Give a clear recommendation and the reason. Be brief."
            ),
            "model_name": MODEL,
            "max_loops": 1,
            "max_tokens": 4000,
        },
        {
            "agent_name": "Synthesizer",
            "description": "Combines the specialists' answers into one brief.",
            "system_prompt": (
                "You combine the expert answers you are given into one short, "
                "decision-oriented brief. Keep every concrete number."
            ),
            "model_name": MODEL,
            "max_loops": 1,
            "max_tokens": 4000,
        },
    ],
    "task": (
        "Should a long-term investor choose SMH or SOXX for semiconductor "
        "exposure? Compare concentration, expense ratio and risk."
    ),
}

response = requests.post(
    f"{BASE_URL}/v1/swarm/completions", headers=headers, json=payload
)
response.raise_for_status()
result = response.json()

# output is the whole conversation, the synthesis is the last entry
for message in result["output"]:
    print(f"[{message['role']}]\n{message['content']}\n")
print(json.dumps(result["usage"], indent=2))
