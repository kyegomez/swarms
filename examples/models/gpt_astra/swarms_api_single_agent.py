"""One gpt-6-astra agent through the Swarms API, no framework install needed.

Run:
    export SWARMS_API_KEY=...   # https://swarms.world/platform/api-keys
    python examples/models/gpt_astra/swarms_api_single_agent.py
"""

import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("SWARMS_API_KEY")
BASE_URL = "https://api.swarms.world"

headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}

payload = {
    "agent_config": {
        "agent_name": "Quantitative-Trading-Agent",
        "description": "Quantitative trading and financial research agent.",
        "system_prompt": (
            "You are a quantitative trading assistant. When comparing "
            "investment options, weigh performance, expense ratio, holdings, "
            "strategy and risk, and state your assumptions."
        ),
        "model_name": "gpt-6-astra",
        "max_loops": 1,
        "max_tokens": 8000,
    },
    "task": (
        "Compare the SMH and SOXX semiconductor ETFs in one short paragraph: "
        "concentration, expense ratio, and which suits a long-term investor."
    ),
}

response = requests.post(
    f"{BASE_URL}/v1/agent/completions", headers=headers, json=payload
)
response.raise_for_status()
result = response.json()

# outputs is the conversation, the agent's answer is the last entry
print(result["outputs"][-1]["content"])
print(json.dumps(result["usage"], indent=2))
