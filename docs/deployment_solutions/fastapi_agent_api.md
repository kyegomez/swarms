# FastAPI Agent API

This guide shows you how to deploy your Swarms agents as REST APIs using FastAPI and Uvicorn. This is the fastest way to expose your agents via HTTP endpoints.

## Overview

FastAPI is a modern, fast web framework for building APIs with Python. Combined with Uvicorn (ASGI server), it provides excellent performance and automatic API documentation.

**Benefits:**
- **Fast**: Built on Starlette and Pydantic
- **Auto-docs**: Automatic OpenAPI/Swagger documentation
- **Type-safe**: Full type hints and validation
- **Easy**: Minimal boilerplate code
- **Monitoring**: Built-in logging and metrics

## Quick Start

### 1. Install Dependencies

```bash
pip install fastapi uvicorn swarms
```

### 2. Create Your Agent API

Create a file called `agent_api.py`:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from swarms import Agent
import uvicorn
from typing import Optional, Dict, Any

# Initialize FastAPI app
app = FastAPI(
    title="Swarms Agent API",
    description="REST API for Swarms agents",
    version="1.0.0"
)

# Pydantic models for request/response
class AgentRequest(BaseModel):
    """Request model for agent tasks"""
    task: str
    agent_name: Optional[str] = "default"
    max_loops: Optional[int] = 1
    temperature: Optional[float] = None

class AgentResponse(BaseModel):
    """Response model for agent tasks"""
    success: bool
    result: str
    agent_name: str
    task: str
    execution_time: Optional[float] = None

# Initialize your agent (you can customize this)
def create_agent(agent_name: str = "default") -> Agent:
    """Create and return a configured agent"""
    return Agent(
        agent_name=agent_name,
        agent_description="Versatile AI agent for various tasks",
        system_prompt="""You are a helpful AI assistant that can handle a wide variety of tasks.
        You provide clear, accurate, and helpful responses while maintaining a professional tone.
        Always strive to be thorough and accurate in your responses.""",
        model_name="claude-sonnet-4-20250514",
        dynamic_temperature_enabled=True,
        max_loops=1,
        dynamic_context_window=True,
    )

# API endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Swarms Agent API is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "service": "Swarms Agent API",
        "version": "1.0.0"
    }

@app.post("/agent/run", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    """Run an agent with the specified task"""
    try:
        import time
        start_time = time.time()
        
        # Create agent instance
        agent = create_agent(request.agent_name)
        
        # Run the agent
        result = agent.run(
            task=request.task,
            max_loops=request.max_loops
        )
        
        execution_time = time.time() - start_time
        
        return AgentResponse(
            success=True,
            result=str(result),
            agent_name=request.agent_name,
            task=request.task,
            execution_time=execution_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")

@app.post("/agent/chat")
async def chat_with_agent(request: AgentRequest):
    """Chat with an agent (conversational mode)"""
    try:
        agent = create_agent(request.agent_name)
        
        # For chat, you might want to maintain conversation history
        # This is a simple implementation
        result = agent.run(
            task=request.task,
            max_loops=request.max_loops
        )
        
        return {
            "success": True,
            "response": str(result),
            "agent_name": request.agent_name
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.get("/agents/available")
async def list_available_agents():
    """List available agent configurations"""
    return {
        "agents": [
            {
                "name": "default",
                "description": "Versatile AI agent for various tasks",
                "model": "claude-sonnet-4-20250514"
            },
            {
                "name": "quantitative-trading",
                "description": "Advanced quantitative trading and algorithmic analysis agent",
                "model": "claude-sonnet-4-20250514"
            }
        ]
    }

# Custom agent endpoint example
@app.post("/agent/quantitative-trading")
async def run_quantitative_trading_agent(request: AgentRequest):
    """Run the quantitative trading agent specifically"""
    try:
        # Create specialized quantitative trading agent
        agent = Agent(
            agent_name="Quantitative-Trading-Agent",
            agent_description="Advanced quantitative trading and algorithmic analysis agent",
            system_prompt="""You are an expert quantitative trading agent with deep expertise in:
            - Algorithmic trading strategies and implementation
            - Statistical arbitrage and market making
            - Risk management and portfolio optimization
            - High-frequency trading systems
            - Market microstructure analysis
            - Quantitative research methodologies
            - Financial mathematics and stochastic processes
            - Machine learning applications in trading""",
            model_name="claude-sonnet-4-20250514",
            dynamic_temperature_enabled=True,
            max_loops=request.max_loops,
            dynamic_context_window=True,
        )
        
        result = agent.run(task=request.task)
        
        return {
            "success": True,
            "result": str(result),
            "agent_name": "Quantitative-Trading-Agent",
            "task": request.task
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quantitative trading agent failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 3. Run Your API

```bash
python agent_api.py
```

Or with uvicorn directly:

```bash
uvicorn agent_api:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Test Your API

Your API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc

## Usage Examples

### Using curl

```bash
# Basic agent task
curl -X POST "http://localhost:8000/agent/run" \
     -H "Content-Type: application/json" \
     -d '{"task": "What are the best top 3 ETFs for gold coverage?"}'

# Quantitative trading agent
curl -X POST "http://localhost:8000/agent/quantitative-trading" \
     -H "Content-Type: application/json" \
     -d '{"task": "Analyze the current market conditions for gold ETFs"}'
```

### Using Python requests

```python
import requests

# Run basic agent
response = requests.post(
    "http://localhost:8000/agent/run",
    json={"task": "Explain quantum computing in simple terms"}
)
print(response.json())

# Run quantitative trading agent
response = requests.post(
    "http://localhost:8000/agent/quantitative-trading",
    json={"task": "What are the key factors affecting gold prices today?"}
)
print(response.json())
```

## Advanced Configuration

### Environment Variables

Create a `.env` file for configuration:

```bash
# .env
AGENT_MODEL_NAME=claude-sonnet-4-20250514
AGENT_MAX_LOOPS=3
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=info
```

### Enhanced Agent Factory

```python
import os
from typing import Dict, Type
from swarms import Agent

class AgentFactory:
    """Factory for creating different types of agents"""
    
    AGENT_CONFIGS = {
        "default": {
            "agent_name": "Default-Agent",
            "agent_description": "Versatile AI agent for various tasks",
            "system_prompt": "You are a helpful AI assistant...",
            "model_name": "claude-sonnet-4-20250514"
        },
        "quantitative-trading": {
            "agent_name": "Quantitative-Trading-Agent",
            "agent_description": "Advanced quantitative trading agent",
            "system_prompt": "You are an expert quantitative trading agent...",
            "model_name": "claude-sonnet-4-20250514"
        },
        "research": {
            "agent_name": "Research-Agent",
            "agent_description": "Academic research and analysis agent",
            "system_prompt": "You are an expert research agent...",
            "model_name": "claude-sonnet-4-20250514"
        }
    }
    
    @classmethod
    def create_agent(cls, agent_type: str = "default", **kwargs) -> Agent:
        """Create an agent of the specified type"""
        if agent_type not in cls.AGENT_CONFIGS:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        config = cls.AGENT_CONFIGS[agent_type].copy()
        config.update(kwargs)
        
        return Agent(**config)
```

### Authentication & Rate Limiting

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import time

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token"""
    # Implement your token verification logic here
    if credentials.credentials != "your-secret-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    return credentials.credentials

@app.post("/agent/run", response_model=AgentResponse)
@limiter.limit("10/minute")
async def run_agent(
    request: AgentRequest,
    token: str = Depends(verify_token)
):
    """Run an agent with authentication and rate limiting"""
    # ... existing code ...
```

## Production Deployment

### Using Gunicorn

```bash
pip install gunicorn
gunicorn agent_api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "agent_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Using Docker Compose

```yaml
version: '3.8'
services:
  agent-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - AGENT_MODEL_NAME=claude-sonnet-4-20250514
    volumes:
      - ./logs:/app/logs
```

## Monitoring & Logging

### Structured Logging

```python
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request, call_next):
    """Log all requests and responses"""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url}")
    
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - {process_time:.2f}s")
    
    return response
```

### Health Checks

```python
@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with agent status"""
    try:
        # Test agent creation
        agent = create_agent()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "agent_status": "available",
            "model_status": "connected"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }
```

## Best Practices

1. **Error Handling**: Always wrap agent execution in try-catch blocks
2. **Validation**: Use Pydantic models for request validation
3. **Rate Limiting**: Implement rate limiting for production APIs
4. **Authentication**: Add proper authentication for sensitive endpoints
5. **Logging**: Log all requests and responses for debugging
6. **Monitoring**: Add health checks and metrics
7. **Testing**: Write tests for your API endpoints
8. **Documentation**: Keep your API documentation up to date

## Troubleshooting

### Common Issues

1. **Port already in use**: Change the port in the uvicorn command
2. **Agent initialization fails**: Check your API keys and model configuration
3. **Memory issues**: Reduce `max_loops` or implement streaming responses
4. **Timeout errors**: Increase timeout settings for long-running tasks

### Performance Tips

1. **Connection pooling**: Reuse agent instances when possible
2. **Async operations**: Use async/await for I/O operations
3. **Caching**: Cache frequently requested responses
4. **Load balancing**: Use multiple worker processes for high traffic

## Next Steps

- [Cron Job Deployment](cron_job_deployment.md) - For scheduled tasks
- [Docker Deployment](docker_deployment.md) - For containerized deployment
- [Kubernetes Deployment](kubernetes_deployment.md) - For orchestrated deployment
- [Cloud Deployment](cloud_deployment.md) - For cloud platforms

Your FastAPI agent API is now ready to handle requests and scale with your needs!
