
# Finance Swarm Example

1. Get your API key from the Swarms API dashboard [HERE](https://swarms.world/platform/api-keys)
2. Create a `.env` file in the root directory and add your API key:

```bash
SWARMS_API_KEY=<your-api-key>
```

3. Create a Python script to create and trigger the financial swarm:


```python
import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

# Retrieve API key securely from .env
API_KEY = os.getenv("SWARMS_API_KEY")
BASE_URL = "https://swarms-api-285321057562.us-east1.run.app"

# Headers for secure API communication
headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}

def create_financial_swarm(equity_data: str):
    """
    Constructs and triggers a full-stack financial swarm consisting of three agents:
    Equity Analyst, Risk Assessor, and Market Advisor.
    Each agent is provided with a comprehensive, detailed system prompt to ensure high reliability.
    """

    payload = {
        "swarm_name": "Enhanced Financial Analysis Swarm",
        "description": "A swarm of agents specialized in performing comprehensive financial analysis, risk assessment, and market recommendations.",
        "agents": [
            {
                "agent_name": "Equity Analyst",
                "description": "Agent specialized in analyzing equities data to provide insights on stock performance and valuation.",
                "system_prompt": (
                    "You are an experienced equity analyst with expertise in financial markets and stock valuation. "
                    "Your role is to analyze the provided equities data, including historical performance, financial statements, and market trends. "
                    "Provide a detailed analysis of the stock's potential, including valuation metrics and growth prospects. "
                    "Consider macroeconomic factors, industry trends, and company-specific news. Your analysis should be clear, actionable, and well-supported by data."
                ),
                "model_name": "openai/gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 4000,
                "temperature": 0.3,
                "auto_generate_prompt": False
            },
            {
                "agent_name": "Risk Assessor",
                "description": "Agent responsible for evaluating the risks associated with equity investments.",
                "system_prompt": (
                    "You are a certified risk management professional with expertise in financial risk assessment. "
                    "Your task is to evaluate the risks associated with the provided equities data, including market risk, credit risk, and operational risk. "
                    "Provide a comprehensive risk analysis, including potential scenarios and their impact on investment performance. "
                    "Your output should be detailed, reliable, and compliant with current risk management standards."
                ),
                "model_name": "openai/gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 3000,
                "temperature": 0.2,
                "auto_generate_prompt": False
            },
            {
                "agent_name": "Market Advisor",
                "description": "Agent dedicated to suggesting investment strategies based on market conditions and equity analysis.",
                "system_prompt": (
                    "You are a knowledgeable market advisor with expertise in investment strategies and portfolio management. "
                    "Based on the analysis provided by the Equity Analyst and the risk assessment, your task is to recommend a comprehensive investment strategy. "
                    "Your suggestions should include asset allocation, diversification strategies, and considerations for market conditions. "
                    "Explain the rationale behind each recommendation and reference relevant market data where applicable. "
                    "Your recommendations should be reliable, detailed, and clearly prioritized based on risk and return."
                ),
                "model_name": "openai/gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 5000,
                "temperature": 0.3,
                "auto_generate_prompt": False
            }
        ],
        "max_loops": 1,
        "swarm_type": "SequentialWorkflow",
        "task": equity_data,
    }

    # Payload includes the equity data as the task to be processed by the swarm

    response = requests.post(
        f"{BASE_URL}/v1/swarm/completions",
        headers=headers,
        json=payload,
    )

    if response.status_code == 200:
        print("Swarm successfully executed!")
        return json.dumps(response.json(), indent=4)
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None


# Example Equity Data for the Swarm to analyze
if __name__ == "__main__":
    equity_data = (
        "Analyze the equity data for Company XYZ, which has shown a 15% increase in revenue over the last quarter, "
        "with a P/E ratio of 20 and a market cap of $1 billion. Consider the current market conditions and potential risks."
    )

    financial_output = create_financial_swarm(equity_data)
    print(financial_output)
```

4. Run the script:

```bash
python financial_swarm.py
```

