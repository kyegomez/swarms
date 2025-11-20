import os
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

from swarms import Agent
from x402.fastapi.middleware import require_payment

# Load environment variables
load_dotenv()


# Custom system prompt for memecoin analysis
MEMECOIN_ANALYSIS_PROMPT = """You are an expert cryptocurrency analyst specializing in memecoin evaluation and risk assessment.

Your role is to provide comprehensive, data-driven analysis of memecoins to help investors make informed decisions.

For each memecoin analysis, you should evaluate:

1. **Market Overview**:
   - Current market cap and trading volume
   - Price trends and volatility
   - Liquidity assessment

2. **Community & Social Sentiment**:
   - Social media presence and engagement
   - Community size and activity
   - Influencer endorsements or warnings
   - Trending status on platforms

3. **Technical Analysis**:
   - Price patterns and trends
   - Support and resistance levels
   - Trading volume analysis
   - Key technical indicators (RSI, MACD, etc.)

4. **Risk Assessment**:
   - Rug pull indicators
   - Liquidity risks
   - Smart contract security
   - Team transparency
   - Concentration of holdings

5. **Investment Recommendation**:
   - Risk level (Low/Medium/High/Extreme)
   - Potential upside and downside
   - Time horizon considerations
   - Position sizing recommendations

Always provide balanced, objective analysis. Emphasize risks clearly and never guarantee returns.
Be honest about the speculative nature of memecoin investments.

Format your analysis in a clear, structured manner with actionable insights.
"""


# Initialize FastAPI app
app = FastAPI(
    title="Memecoin Analysis Agent",
    description="AI-powered memecoin analysis service monetized with x402",
    version="1.0.0",
)


def create_memecoin_agent(
    agent_name: str = "Memecoin-Analyzer",
    model_name: str = "gpt-4o",
    max_loops: int = 1,
    temperature: float = 0.5,
) -> Agent:
    """
    Create a specialized agent for memecoin analysis.

    Args:
        agent_name: Name identifier for the agent
        model_name: The LLM model to use for analysis
        max_loops: Maximum number of reasoning loops
        temperature: Temperature for response generation (0.0-1.0)

    Returns:
        Configured Agent instance for memecoin analysis
    """
    agent = Agent(
        agent_name=agent_name,
        system_prompt=MEMECOIN_ANALYSIS_PROMPT,
        model_name=model_name,
        max_loops=max_loops,
        autosave=True,
        dashboard=False,
        verbose=True,
        dynamic_temperature_enabled=True,
        temperature=temperature,
    )
    return agent


# Initialize the memecoin analysis agent
memecoin_agent = create_memecoin_agent()


# Request model
class MemecoinRequest(BaseModel):
    """
    Request model for memecoin analysis.

    Attributes:
        symbol: The memecoin ticker symbol (e.g., 'DOGE', 'PEPE', 'SHIB')
        include_social: Whether to include social media sentiment analysis
        include_technicals: Whether to include technical analysis
    """

    symbol: str = Field(..., description="Memecoin ticker symbol")
    include_social: bool = Field(
        default=True, description="Include social media sentiment"
    )
    include_technicals: bool = Field(
        default=True, description="Include technical analysis"
    )


# Apply x402 payment middleware to the analysis endpoint
# For testnet (Base Sepolia), use the free facilitator
# For mainnet, see configuration in README.md
app.middleware("http")(
    require_payment(
        path="/analyze-memecoin",
        price="$0.10",  # 10 cents per analysis
        pay_to_address=os.getenv(
            "WALLET_ADDRESS", "0xYourWalletAddress"
        ),
        network_id="base-sepolia",  # Use "base" for mainnet
        description="Get comprehensive AI-powered memecoin analysis including market sentiment, risk assessment, and investment recommendations",
        input_schema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Memecoin ticker symbol (e.g., DOGE, PEPE, SHIB)",
                },
                "include_social": {
                    "type": "boolean",
                    "description": "Include social media sentiment analysis",
                    "default": True,
                },
                "include_technicals": {
                    "type": "boolean",
                    "description": "Include technical analysis indicators",
                    "default": True,
                },
            },
            "required": ["symbol"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "analysis": {"type": "string"},
                "risk_level": {"type": "string"},
                "recommendation": {"type": "string"},
                "timestamp": {"type": "string"},
            },
        },
    )
)


@app.get("/")
async def root() -> Dict[str, str]:
    """
    Root endpoint providing API information.

    Returns:
        Dictionary with API details and usage instructions
    """
    return {
        "service": "Memecoin Analysis Agent",
        "version": "1.0.0",
        "description": "AI-powered memecoin analysis monetized with x402",
        "endpoints": {
            "/analyze-memecoin": "POST - Analyze a memecoin (requires payment)",
            "/health": "GET - Health check endpoint (free)",
        },
        "pricing": "$0.10 per analysis",
        "network": "base-sepolia (testnet)",
    }


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint (free, no payment required).

    Returns:
        Dictionary with service status
    """
    return {"status": "healthy", "agent": "ready"}


@app.post("/analyze-memecoin")
async def analyze_memecoin(
    request: MemecoinRequest,
) -> Dict[str, Any]:
    """
    Analyze a memecoin using the AI agent (requires x402 payment).

    This endpoint is protected by x402 payment middleware. Users must
    pay $0.10 in USDC to access the analysis.

    Args:
        request: MemecoinRequest with symbol and analysis options

    Returns:
        Dictionary containing comprehensive memecoin analysis
    """
    # Build the analysis query
    query_parts = [f"Analyze the memecoin {request.symbol.upper()}."]

    if request.include_social:
        query_parts.append(
            "Include detailed social media sentiment analysis."
        )

    if request.include_technicals:
        query_parts.append(
            "Include comprehensive technical analysis with key indicators."
        )

    query_parts.append(
        "Provide a clear risk assessment and investment recommendation."
    )

    query = " ".join(query_parts)

    # Run the agent analysis
    analysis_result = memecoin_agent.run(query)

    # Return structured response
    return {
        "symbol": request.symbol.upper(),
        "analysis": analysis_result,
        "risk_level": "See analysis for details",
        "recommendation": "See analysis for details",
        "timestamp": "2025-10-29",
        "disclaimer": "This analysis is for informational purposes only. Not financial advice.",
    }


@app.post("/batch-analyze")
async def batch_analyze(symbols: list[str]) -> Dict[str, Any]:
    """
    Analyze multiple memecoins in a batch (free endpoint for demo).

    Args:
        symbols: List of memecoin ticker symbols

    Returns:
        Dictionary with analysis for multiple coins
    """
    results = {}
    for symbol in symbols[:3]:  # Limit to 3 for demo
        query = (
            f"Provide a brief overview of {symbol.upper()} memecoin."
        )
        analysis = memecoin_agent.run(query)
        results[symbol.upper()] = {
            "brief_analysis": analysis[:200] + "...",
            "full_analysis_endpoint": "/analyze-memecoin",
        }

    return {
        "batch_results": results,
        "note": "For full detailed analysis, use /analyze-memecoin (requires payment)",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
