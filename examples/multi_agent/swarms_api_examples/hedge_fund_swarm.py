# tools - search, code executor, create api

import os
import requests
from dotenv import load_dotenv
import json
from swarms_tools import coin_gecko_coin_api

load_dotenv()

API_KEY = os.getenv("SWARMS_API_KEY")
BASE_URL = "https://api.swarms.world"

headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}

data = coin_gecko_coin_api("bitcoin")

print(data)


def run_health_check():
    response = requests.get(f"{BASE_URL}/health", headers=headers)
    return response.json()


def run_single_swarm():
    payload = {
        "name": "Hedge Fund Analysis Swarm",
        "description": "A highly customized swarm for hedge fund analysis, focusing on market trends, risk assessment, and investment strategies.",
        "agents": [
            {
                "agent_name": "Hedge Fund Analyst",
                "description": "Analyzes market trends and investment opportunities.",
                "system_prompt": "You are a hedge fund analyst with expertise in cryptocurrency. Analyze current market conditions for Bitcoin and major cryptocurrencies. Identify investment opportunities by evaluating volatility and performance. Provide a report with technical and fundamental analysis.",
                "model_name": "openai/gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
            },
            {
                "agent_name": "Risk Assessment Agent",
                "description": "Evaluates risks in investment strategies.",
                "system_prompt": "You are a risk assessment expert in cryptocurrency. Identify and evaluate risks related to investment strategies, including market and credit risks. Provide a risk analysis report with assessments and mitigation strategies.",
                "model_name": "gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
            },
            {
                "agent_name": "Portfolio Manager",
                "description": "Manages and optimizes investment portfolios.",
                "system_prompt": "You are a portfolio manager for a crypto hedge fund. Optimize asset allocation based on market conditions. Analyze existing assets, suggest adjustments, and provide diversification strategies.",
                "model_name": "gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
            },
            {
                "agent_name": "Market Sentiment Analyst",
                "description": "Analyzes market sentiment for trading strategies.",
                "system_prompt": "You are a market sentiment analyst in cryptocurrency. Assess current sentiment by analyzing news and social media. Provide insights on how sentiment impacts investment decisions and summarize key indicators.",
                "model_name": "gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
            },
        ],
        "max_loops": 1,
        "swarm_type": "ConcurrentWorkflow",
        "task": "Analyze Bitcoin right now and provide a detailed report on the current market conditions, including technical and fundamental analysis, and then suggest potential trades with buy and sell recommendations based on the analysis",
        "output_type": "dict",
    }

    response = requests.post(
        f"{BASE_URL}/v1/swarm/completions",
        headers=headers,
        json=payload,
    )

    # return response.json()
    print(response.json())
    print(response.status_code)
    output = response.json()

    return json.dumps(output, indent=4)


if __name__ == "__main__":
    result = run_single_swarm()
    print("Swarm Result:")
    print(result)
