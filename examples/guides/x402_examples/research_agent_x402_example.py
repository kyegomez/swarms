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
