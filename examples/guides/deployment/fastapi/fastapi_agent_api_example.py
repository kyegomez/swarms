"""
FastAPI Agent API Example

This example shows how to deploy your Swarms agents as a REST API using FastAPI and Uvicorn.
Based on the quantitative trading agent from example.py.

Run this file to start your agent API server.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from swarms import Agent
import uvicorn
import time
from typing import Optional, List
from loguru import logger


# Initialize FastAPI app
app = FastAPI(
    title="Swarms Agent API",
    description="REST API for Swarms agents - Quantitative Trading Example",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# Pydantic models for request/response
class AgentRequest(BaseModel):
    """Request model for agent tasks"""

    task: str
    agent_name: Optional[str] = "quantitative-trading"
    max_loops: Optional[int] = 1
    temperature: Optional[float] = None


class AgentResponse(BaseModel):
    """Response model for agent tasks"""

    success: bool
    result: str
    agent_name: str
    task: str
    execution_time: Optional[float] = None
    timestamp: str


class AgentInfo(BaseModel):
    """Model for agent information"""

    name: str
    description: str
    model: str
    capabilities: List[str]


# Agent configurations
AGENT_CONFIGS = {
    "quantitative-trading": {
        "agent_name": "Quantitative-Trading-Agent",
        "agent_description": "Advanced quantitative trading and algorithmic analysis agent",
        "system_prompt": """You are an expert quantitative trading agent with deep expertise in:
        - Algorithmic trading strategies and implementation
        - Statistical arbitrage and market making
        - Risk management and portfolio optimization
        - High-frequency trading systems
        - Market microstructure analysis
        - Quantitative research methodologies
        - Financial mathematics and stochastic processes
        - Machine learning applications in trading
        
        Your core responsibilities include:
        1. Developing and backtesting trading strategies
        2. Analyzing market data and identifying alpha opportunities
        3. Implementing risk management frameworks
        4. Optimizing portfolio allocations
        5. Conducting quantitative research
        6. Monitoring market microstructure
        7. Evaluating trading system performance
        
        You maintain strict adherence to:
        - Mathematical rigor in all analyses
        - Statistical significance in strategy development
        - Risk-adjusted return optimization
        - Market impact minimization
        - Regulatory compliance
        - Transaction cost analysis
        - Performance attribution
        
        You communicate in precise, technical terms while maintaining clarity for stakeholders.""",
        "model_name": "claude-sonnet-4-20250514",
        "capabilities": [
            "Trading strategy development",
            "Market analysis",
            "Risk management",
            "Portfolio optimization",
            "Quantitative research",
            "Performance evaluation",
        ],
    },
    "general": {
        "agent_name": "General-AI-Agent",
        "agent_description": "Versatile AI agent for various tasks",
        "system_prompt": """You are a helpful AI assistant that can handle a wide variety of tasks.
        You provide clear, accurate, and helpful responses while maintaining a professional tone.
        Always strive to be thorough and accurate in your responses.""",
        "model_name": "claude-sonnet-4-20250514",
        "capabilities": [
            "General assistance",
            "Problem solving",
            "Information gathering",
            "Analysis and synthesis",
        ],
    },
}


def create_agent(
    agent_type: str = "quantitative-trading", **kwargs
) -> Agent:
    """Create and return a configured agent"""
    if agent_type not in AGENT_CONFIGS:
        raise ValueError(f"Unknown agent type: {agent_type}")

    config = AGENT_CONFIGS[agent_type].copy()
    agent_config = {
        "agent_name": config["agent_name"],
        "agent_description": config["agent_description"],
        "system_prompt": config["system_prompt"],
        "model_name": config["model_name"],
        "dynamic_temperature_enabled": True,
        "max_loops": kwargs.get("max_loops", 1),
        "dynamic_context_window": True,
    }

    # Update with any additional kwargs
    agent_config.update(kwargs)

    return Agent(**agent_config)


# API endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Swarms Agent API is running!",
        "status": "healthy",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "service": "Swarms Agent API",
        "version": "1.0.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/agents", response_model=List[AgentInfo])
async def list_available_agents():
    """List available agent configurations"""
    agents = []
    for agent_type, config in AGENT_CONFIGS.items():
        agents.append(
            AgentInfo(
                name=agent_type,
                description=config["agent_description"],
                model=config["model_name"],
                capabilities=config["capabilities"],
            )
        )
    return agents


@app.post("/agent/run", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    """Run an agent with the specified task"""
    try:
        start_time = time.time()

        # Create agent instance
        agent = create_agent(
            agent_type=request.agent_name, max_loops=request.max_loops
        )

        # Run the agent
        result = agent.run(task=request.task)

        execution_time = time.time() - start_time

        logger.info(
            f"Agent {request.agent_name} completed task in {execution_time:.2f}s"
        )

        return AgentResponse(
            success=True,
            result=str(result),
            agent_name=request.agent_name,
            task=request.task,
            execution_time=execution_time,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    except Exception as e:
        logger.error(f"Agent execution failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent execution failed: {str(e)}",
        )


@app.post("/agent/chat")
async def chat_with_agent(request: AgentRequest):
    """Chat with an agent (conversational mode)"""
    try:
        agent = create_agent(
            agent_type=request.agent_name, max_loops=request.max_loops
        )

        result = agent.run(task=request.task)

        return {
            "success": True,
            "response": str(result),
            "agent_name": request.agent_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    except Exception as e:
        logger.error(f"Chat failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Chat failed: {str(e)}"
        )


@app.post("/agent/quantitative-trading")
async def run_quantitative_trading_agent(request: AgentRequest):
    """Run the quantitative trading agent specifically"""
    try:
        start_time = time.time()

        # Create specialized quantitative trading agent
        agent = create_agent(
            "quantitative-trading", max_loops=request.max_loops
        )

        result = agent.run(task=request.task)

        execution_time = time.time() - start_time

        logger.info(
            f"Quantitative trading agent completed task in {execution_time:.2f}s"
        )

        return {
            "success": True,
            "result": str(result),
            "agent_name": "Quantitative-Trading-Agent",
            "task": request.task,
            "execution_time": execution_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    except Exception as e:
        logger.error(f"Quantitative trading agent failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Quantitative trading agent failed: {str(e)}",
        )


@app.get("/examples")
async def get_example_requests():
    """Get example API requests for testing"""
    return {
        "examples": [
            {
                "endpoint": "/agent/run",
                "method": "POST",
                "description": "Run any available agent",
                "request_body": {
                    "task": "What are the best top 3 ETFs for gold coverage?",
                    "agent_name": "quantitative-trading",
                    "max_loops": 1,
                },
            },
            {
                "endpoint": "/agent/quantitative-trading",
                "method": "POST",
                "description": "Run quantitative trading agent specifically",
                "request_body": {
                    "task": "Analyze the current market conditions for gold ETFs",
                    "max_loops": 1,
                },
            },
            {
                "endpoint": "/agent/chat",
                "method": "POST",
                "description": "Chat with an agent",
                "request_body": {
                    "task": "Explain the concept of statistical arbitrage",
                    "agent_name": "quantitative-trading",
                },
            },
        ]
    }


# Middleware for logging
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all requests and responses"""
    start_time = time.time()

    # Log request
    logger.info(f"Request: {request.method} {request.url}")

    response = await call_next(request)

    # Log response
    process_time = time.time() - start_time
    logger.info(
        f"Response: {response.status_code} - {process_time:.2f}s"
    )

    return response


if __name__ == "__main__":
    print("Starting Swarms Agent API...")
    print(
        "API Documentation will be available at: http://localhost:8000/docs"
    )
    print("Alternative docs at: http://localhost:8000/redoc")
    print("Health check at: http://localhost:8000/health")
    print("Available agents at: http://localhost:8000/agents")
    print("Examples at: http://localhost:8000/examples")
    print("\n" + "=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
