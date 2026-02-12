# X402 Payment Integration with Swarms Agents

X402 is a protocol that enables seamless cryptocurrency payments for API endpoints. This guide demonstrates how to monetize your Swarms agents by integrating X402 payment requirements into your FastAPI applications.

With X402, you can:

| Feature                                             | Description                                  |
|-----------------------------------------------------|----------------------------------------------|
| Charge per API request                              | Monetize your agents on a per-call basis     |
| Accept cryptocurrency payments                      | e.g., Base, Base Sepolia, and more           |
| Payment gate protection for agent endpoints         | Secure endpoints with pay-to-access gates    |
| Create pay-per-use AI services                      | Offer AI agents as on-demand paid services   |

## Prerequisites

Before you begin, ensure you have:

- Python 3.10 or higher
- A cryptocurrency wallet address (for receiving payments)
- API keys for your AI model provider (e.g., OpenAI)
- An Exa API key (if using web search functionality)

## Installation

Install the required dependencies:

```bash
pip install swarms x402 fastapi uvicorn python-dotenv swarms-tools
```

## Environment Setup

Create a `.env` file in your project root:

```bash
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Exa API Key (for web search)
EXA_API_KEY=your_exa_api_key_here

# Your wallet address (where you'll receive payments)
WALLET_ADDRESS=0xYourWalletAddressHere
```

## Basic X402 Integration Example

Here's a complete example of a research agent with X402 payment integration:

```python
from dotenv import load_dotenv
from fastapi import FastAPI
from swarms_tools import exa_search

from swarms import Agent
from x402.fastapi.middleware import require_payment

# Load environment variables
load_dotenv()

app = FastAPI(title="Research Agent API")

# Initialize the research agent
research_agent = Agent(
    agent_name="Research-Agent",
    system_prompt="You are an expert research analyst. Conduct thorough research on the given topic and provide comprehensive, well-structured insights with citations.",
    model_name="gpt-4o-mini",
    max_loops=1,
    tools=[exa_search],
)


# Apply x402 payment middleware to the research endpoint
app.middleware("http")(
    require_payment(
        path="/research",
        price="$0.01",
        pay_to_address="0xYourWalletAddressHere",
        network_id="base-sepolia",
        description="AI-powered research agent that conducts comprehensive research on any topic",
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Research topic or question",
                }
            },
            "required": ["query"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "research": {
                    "type": "string",
                    "description": "Comprehensive research results",
                }
            },
        },
    )
)


@app.get("/research")
async def conduct_research(query: str):
    """
    Conduct research on a given topic using the research agent.

    Args:
        query: The research topic or question

    Returns:
        Research results from the agent
    """
    result = research_agent.run(query)
    return {"research": result}


@app.get("/")
async def root():
    """Health check endpoint (free, no payment required)"""
    return {
        "message": "Research Agent API with x402 payments",
        "endpoints": {
            "/research": "Paid endpoint - $0.01 per request",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
```


## Running Your Service

Start the server:

```bash
python research_agent_x402_example.py
```

Or with uvicorn directly:

```bash
uvicorn research_agent_x402_example:app --host 0.0.0.0 --port 8000 --reload
```

Your API will be available at:

- Main endpoint: `http://localhost:8000/`

- Research endpoint: `http://localhost:8000/research`

- API docs: `http://localhost:8000/docs`


## Next Steps

1. Experiment with different pricing models
2. Add multiple agents with specialized capabilities
3. Implement analytics to track usage and revenue
4. Add per-agent spending limits and circuit breakers (see [X402 Spending Limits](x402_spending_limits.md))
5. Deploy to production (see [Deployment Solutions](../deployment_solutions/overview.md))
6. Integrate with your existing payment processing
