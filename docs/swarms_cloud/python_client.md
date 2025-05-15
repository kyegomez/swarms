# Swarms API Client Reference Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Authentication](#authentication)
5. [Client Configuration](#client-configuration)
6. [API Endpoints Overview](#api-endpoints-overview)
7. [Core Methods](#core-methods)
8. [Swarm Management](#swarm-management)
9. [Agent Management](#agent-management)
10. [Batch Operations](#batch-operations)
11. [Health and Monitoring](#health-and-monitoring)
12. [Error Handling](#error-handling)
13. [Performance Optimization](#performance-optimization)
14. [Type Reference](#type-reference)
15. [Code Examples](#code-examples)
16. [Best Practices](#best-practices)
17. [Troubleshooting](#troubleshooting)

## Introduction

The Swarms API Client is a production-grade Python library designed to interact with the Swarms API. It provides both synchronous and asynchronous interfaces for maximum flexibility, enabling developers to create and manage swarms of AI agents efficiently. The client includes advanced features such as automatic retrying, response caching, connection pooling, and comprehensive error handling.

### Key Features

- **Dual Interface**: Both synchronous and asynchronous APIs
- **Automatic Retrying**: Built-in retry logic with exponential backoff
- **Response Caching**: TTL-based caching for improved performance
- **Connection Pooling**: Optimized connection management
- **Type Safety**: Pydantic models for input validation
- **Comprehensive Logging**: Structured logging with Loguru
- **Thread-Safe**: Safe for use in multi-threaded applications
- **Rate Limiting**: Built-in rate limit handling
- **Performance Optimized**: DNS caching, TCP optimizations, and more

## Installation

```bash
pip install swarms-client
```


## Quick Start

```python
from swarms_client import SwarmsClient

# Initialize the client
client = SwarmsClient(api_key="your-api-key")

# Create a simple swarm
swarm = client.create_swarm(
    name="analysis-swarm",
    task="Analyze this market data",
    agents=[
        {
            "agent_name": "data-analyst",
            "model_name": "gpt-4",
            "role": "worker"
        }
    ]
)

# Run a single agent
result = client.run_agent(
    agent_name="researcher",
    task="Research the latest AI trends",
    model_name="gpt-4"
)
```

### Async Example

```python
import asyncio
from swarms_client import SwarmsClient

async def main():
    async with SwarmsClient(api_key="your-api-key") as client:
        # Create a swarm asynchronously
        swarm = await client.async_create_swarm(
            name="async-swarm",
            task="Process these documents",
            agents=[
                {
                    "agent_name": "document-processor",
                    "model_name": "gpt-4",
                    "role": "worker"
                }
            ]
        )
        print(swarm)

asyncio.run(main())
```

## Authentication

### Obtaining API Keys

API keys can be obtained from the Swarms platform at: [https://swarms.world/platform/api-keys](https://swarms.world/platform/api-keys)

### Setting API Keys

There are three ways to provide your API key:

1. **Direct Parameter** (Recommended for development):
```python
client = SwarmsClient(api_key="your-api-key")
```

2. **Environment Variable** (Recommended for production):
```bash
export SWARMS_API_KEY="your-api-key"
```
```python
client = SwarmsClient()  # Will use SWARMS_API_KEY env var
```

3. **Configuration Object**:
```python
from swarms_client.config import SwarmsConfig

SwarmsConfig.set_api_key("your-api-key")
client = SwarmsClient()
```

## Client Configuration

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | Optional[str] | None | API key for authentication |
| `base_url` | Optional[str] | "https://api.swarms.world" | Base URL for the API |
| `timeout` | Optional[int] | 30 | Request timeout in seconds |
| `max_retries` | Optional[int] | 3 | Maximum number of retry attempts |
| `max_concurrent_requests` | Optional[int] | 100 | Maximum concurrent requests |
| `retry_on_status` | Optional[Set[int]] | {429, 500, 502, 503, 504} | HTTP status codes to retry |
| `retry_delay` | Optional[float] | 1.0 | Initial retry delay in seconds |
| `max_retry_delay` | Optional[int] | 60 | Maximum retry delay in seconds |
| `jitter` | bool | True | Add random jitter to retry delays |
| `enable_cache` | bool | True | Enable response caching |
| `thread_pool_size` | Optional[int] | min(32, max_concurrent_requests * 2) | Thread pool size for sync operations |

### Configuration Example

```python
from swarms_client import SwarmsClient

client = SwarmsClient(
    api_key="your-api-key",
    base_url="https://api.swarms.world",
    timeout=60,
    max_retries=5,
    max_concurrent_requests=50,
    retry_delay=2.0,
    enable_cache=True,
    thread_pool_size=20
)
```

## API Endpoints Overview

### Endpoint Reference Table

| Endpoint | Method | Description | Sync Method | Async Method |
|----------|--------|-------------|-------------|--------------|
| `/health` | GET | Check API health | `get_health()` | `async_get_health()` |
| `/v1/swarm/completions` | POST | Create and run a swarm | `create_swarm()` | `async_create_swarm()` |
| `/v1/swarm/{swarm_id}/run` | POST | Run existing swarm | `run_swarm()` | `async_run_swarm()` |
| `/v1/swarm/{swarm_id}/logs` | GET | Get swarm logs | `get_swarm_logs()` | `async_get_swarm_logs()` |
| `/v1/models/available` | GET | List available models | `get_available_models()` | `async_get_available_models()` |
| `/v1/swarms/available` | GET | List swarm types | `get_swarm_types()` | `async_get_swarm_types()` |
| `/v1/agent/completions` | POST | Run single agent | `run_agent()` | `async_run_agent()` |
| `/v1/agent/batch/completions` | POST | Run agent batch | `run_agent_batch()` | `async_run_agent_batch()` |
| `/v1/swarm/batch/completions` | POST | Run swarm batch | `run_swarm_batch()` | `async_run_swarm_batch()` |
| `/v1/swarm/logs` | GET | Get API logs | `get_api_logs()` | `async_get_api_logs()` |

## Core Methods

### Health Check

Check the API health status to ensure the service is operational.

```python
# Synchronous
health = client.get_health()

# Asynchronous
health = await client.async_get_health()
```

**Response Example:**
```json
{
    "status": "healthy",
    "version": "1.0.0",
    "timestamp": "2025-01-20T12:00:00Z"
}
```

### Available Models

Retrieve a list of all available models that can be used with agents.

```python
# Synchronous
models = client.get_available_models()

# Asynchronous
models = await client.async_get_available_models()
```

**Response Example:**
```json
{
    "models": [
        "gpt-4",
        "gpt-3.5-turbo",
        "claude-3-opus",
        "claude-3-sonnet"
    ]
}
```

### Swarm Types

Get available swarm architecture types.

```python
# Synchronous
swarm_types = client.get_swarm_types()

# Asynchronous
swarm_types = await client.async_get_swarm_types()
```

**Response Example:**
```json
{
    "swarm_types": [
        "sequential",
        "parallel",
        "hierarchical",
        "mesh"
    ]
}
```

## Swarm Management

### Create Swarm

Create and run a new swarm with specified configuration.

#### Method Signature

```python
def create_swarm(
    self,
    name: str,
    task: str,
    agents: List[AgentSpec],
    description: Optional[str] = None,
    max_loops: int = 1,
    swarm_type: Optional[str] = None,
    rearrange_flow: Optional[str] = None,
    return_history: bool = True,
    rules: Optional[str] = None,
    tasks: Optional[List[str]] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    stream: bool = False,
    service_tier: str = "standard",
) -> Dict[str, Any]
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `name` | str | Yes | - | Name of the swarm |
| `task` | str | Yes | - | Main task for the swarm |
| `agents` | List[AgentSpec] | Yes | - | List of agent specifications |
| `description` | Optional[str] | No | None | Swarm description |
| `max_loops` | int | No | 1 | Maximum execution loops |
| `swarm_type` | Optional[str] | No | None | Type of swarm architecture |
| `rearrange_flow` | Optional[str] | No | None | Flow rearrangement instructions |
| `return_history` | bool | No | True | Whether to return execution history |
| `rules` | Optional[str] | No | None | Swarm behavior rules |
| `tasks` | Optional[List[str]] | No | None | List of subtasks |
| `messages` | Optional[List[Dict]] | No | None | Initial messages |
| `stream` | bool | No | False | Whether to stream output |
| `service_tier` | str | No | "standard" | Service tier for processing |

#### Example

```python
from swarms_client.models import AgentSpec

# Define agents
agents = [
    AgentSpec(
        agent_name="researcher",
        model_name="gpt-4",
        role="leader",
        system_prompt="You are an expert researcher.",
        temperature=0.7,
        max_tokens=1000
    ),
    AgentSpec(
        agent_name="analyst",
        model_name="gpt-3.5-turbo",
        role="worker",
        system_prompt="You are a data analyst.",
        temperature=0.5,
        max_tokens=800
    )
]

# Create swarm
swarm = client.create_swarm(
    name="research-team",
    task="Research and analyze climate change impacts",
    agents=agents,
    description="A swarm for climate research",
    max_loops=3,
    swarm_type="hierarchical",
    rules="Always cite sources and verify facts"
)
```

### Run Swarm

Run an existing swarm by its ID.

```python
# Synchronous
result = client.run_swarm(swarm_id="swarm-123")

# Asynchronous
result = await client.async_run_swarm(swarm_id="swarm-123")
```

### Get Swarm Logs

Retrieve execution logs for a specific swarm.

```python
# Synchronous
logs = client.get_swarm_logs(swarm_id="swarm-123")

# Asynchronous
logs = await client.async_get_swarm_logs(swarm_id="swarm-123")
```

**Response Example:**
```json
{
    "logs": [
        {
            "timestamp": "2025-01-20T12:00:00Z",
            "level": "INFO",
            "message": "Swarm started",
            "agent": "researcher",
            "task": "Initial research"
        }
    ]
}
```

## Agent Management

### Run Agent

Run a single agent with specified configuration.

#### Method Signature

```python
def run_agent(
    self,
    agent_name: str,
    task: str,
    model_name: str = "gpt-4",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    system_prompt: Optional[str] = None,
    description: Optional[str] = None,
    auto_generate_prompt: bool = False,
    role: str = "worker",
    max_loops: int = 1,
    tools_dictionary: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `agent_name` | str | Yes | - | Name of the agent |
| `task` | str | Yes | - | Task for the agent |
| `model_name` | str | No | "gpt-4" | Model to use |
| `temperature` | float | No | 0.7 | Generation temperature |
| `max_tokens` | int | No | 1000 | Maximum tokens |
| `system_prompt` | Optional[str] | No | None | System prompt |
| `description` | Optional[str] | No | None | Agent description |
| `auto_generate_prompt` | bool | No | False | Auto-generate prompts |
| `role` | str | No | "worker" | Agent role |
| `max_loops` | int | No | 1 | Maximum loops |
| `tools_dictionary` | Optional[List[Dict]] | No | None | Available tools |

#### Example

```python
# Run a single agent
result = client.run_agent(
    agent_name="code-reviewer",
    task="Review this Python code for best practices",
    model_name="gpt-4",
    temperature=0.3,
    max_tokens=1500,
    system_prompt="You are an expert Python developer.",
    role="expert"
)

# With tools
tools = [
    {
        "name": "code_analyzer",
        "description": "Analyze code quality",
        "parameters": {
            "language": "python",
            "metrics": ["complexity", "coverage"]
        }
    }
]

result = client.run_agent(
    agent_name="analyzer",
    task="Analyze this codebase",
    tools_dictionary=tools
)
```

## Batch Operations

### Run Agent Batch

Run multiple agents in parallel for improved efficiency.

```python
# Define multiple agent configurations
agents = [
    {
        "agent_name": "agent1",
        "task": "Task 1",
        "model_name": "gpt-4"
    },
    {
        "agent_name": "agent2",
        "task": "Task 2",
        "model_name": "gpt-3.5-turbo"
    }
]

# Run batch
results = client.run_agent_batch(agents=agents)
```

### Run Swarm Batch

Run multiple swarms in parallel.

```python
# Define multiple swarm configurations
swarms = [
    {
        "name": "swarm1",
        "task": "Research topic A",
        "agents": [{"agent_name": "researcher1", "model_name": "gpt-4"}]
    },
    {
        "name": "swarm2",
        "task": "Research topic B",
        "agents": [{"agent_name": "researcher2", "model_name": "gpt-4"}]
    }
]

# Run batch
results = client.run_swarm_batch(swarms=swarms)
```

## Health and Monitoring

### API Logs

Retrieve all API request logs for your API key.

```python
# Synchronous
logs = client.get_api_logs()

# Asynchronous
logs = await client.async_get_api_logs()
```

**Response Example:**
```json
{
    "logs": [
        {
            "request_id": "req-123",
            "timestamp": "2025-01-20T12:00:00Z",
            "method": "POST",
            "endpoint": "/v1/agent/completions",
            "status": 200,
            "duration_ms": 1234
        }
    ]
}
```

## Error Handling

### Exception Types

| Exception | Description | Common Causes |
|-----------|-------------|---------------|
| `SwarmsError` | Base exception | General API errors |
| `AuthenticationError` | Authentication failed | Invalid API key |
| `RateLimitError` | Rate limit exceeded | Too many requests |
| `ValidationError` | Input validation failed | Invalid parameters |
| `APIError` | API returned an error | Server-side issues |

### Error Handling Example

```python
from swarms_client import (
    SwarmsClient,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    APIError
)

try:
    result = client.run_agent(
        agent_name="test",
        task="Analyze data"
    )
except AuthenticationError:
    print("Invalid API key. Please check your credentials.")
except RateLimitError:
    print("Rate limit exceeded. Please wait before retrying.")
except ValidationError as e:
    print(f"Invalid input: {e}")
except APIError as e:
    print(f"API error: {e.message} (Status: {e.status_code})")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Optimization

### Caching

The client includes built-in response caching for GET requests:

```python
# Enable caching (default)
client = SwarmsClient(api_key="your-key", enable_cache=True)

# Disable caching
client = SwarmsClient(api_key="your-key", enable_cache=False)

# Skip cache for specific request
health = await client.async_get_health(skip_cache=True)
```

### Connection Pooling

The client automatically manages connection pools for optimal performance:

```python
# Configure pool size
client = SwarmsClient(
    api_key="your-key",
    max_concurrent_requests=50,  # Pool size
    thread_pool_size=20          # Thread pool for sync ops
)
```

### Batch Operations

Use batch operations for processing multiple items:

```python
# Instead of this (sequential)
results = []
for task in tasks:
    result = client.run_agent(agent_name="agent", task=task)
    results.append(result)

# Do this (parallel)
agents = [{"agent_name": "agent", "task": task} for task in tasks]
results = client.run_agent_batch(agents=agents)
```

## Type Reference

### AgentSpec

```python
class AgentSpec(BaseModel):
    agent_name: str
    system_prompt: Optional[str] = None
    description: Optional[str] = None
    model_name: str = "gpt-4"
    auto_generate_prompt: bool = False
    max_tokens: int = 1000
    temperature: float = 0.5
    role: Literal["worker", "leader", "expert"] = "worker"
    max_loops: int = 1
    tools_dictionary: Optional[List[Dict[str, Any]]] = None
```

### SwarmSpec

```python
class SwarmSpec(BaseModel):
    name: str
    description: Optional[str] = None
    agents: List[AgentSpec]
    swarm_type: Optional[str] = None
    rearrange_flow: Optional[str] = None
    task: str
    return_history: bool = True
    rules: Optional[str] = None
    tasks: Optional[List[str]] = None
    messages: Optional[List[Dict[str, Any]]] = None
    max_loops: int = 1
    stream: bool = False
    service_tier: Literal["standard", "premium"] = "standard"
```

### AgentCompletion

```python
class AgentCompletion(BaseModel):
    agent_config: AgentSpec
    task: str
```

## Code Examples

### Complete Data Analysis Swarm

```python
from swarms_client import SwarmsClient
from swarms_client.models import AgentSpec

# Initialize client
client = SwarmsClient(api_key="your-api-key")

# Define specialized agents
agents = [
    AgentSpec(
        agent_name="data-collector",
        model_name="gpt-4",
        role="worker",
        system_prompt="You collect and organize data from various sources.",
        temperature=0.3,
        max_tokens=1000
    ),
    AgentSpec(
        agent_name="statistician",
        model_name="gpt-4",
        role="worker",
        system_prompt="You perform statistical analysis on data.",
        temperature=0.2,
        max_tokens=1500
    ),
    AgentSpec(
        agent_name="report-writer",
        model_name="gpt-4",
        role="leader",
        system_prompt="You create comprehensive reports from analysis.",
        temperature=0.7,
        max_tokens=2000
    )
]

# Create and run swarm
swarm = client.create_swarm(
    name="data-analysis-swarm",
    task="Analyze sales data and create quarterly report",
    agents=agents,
    swarm_type="sequential",
    max_loops=2,
    rules="Always include statistical significance in analysis"
)

print(f"Analysis complete: {swarm['result']}")
```

### Async Web Scraping System

```python
import asyncio
from swarms_client import SwarmsClient

async def scrape_and_analyze(urls):
    async with SwarmsClient(api_key="your-api-key") as client:
        # Run scrapers in parallel
        scraper_tasks = []
        for i, url in enumerate(urls):
            task = client.async_run_agent(
                agent_name=f"scraper-{i}",
                task=f"Extract main content from {url}",
                model_name="gpt-3.5-turbo",
                temperature=0.1
            )
            scraper_tasks.append(task)
        
        # Wait for all scrapers
        scraped_data = await asyncio.gather(*scraper_tasks)
        
        # Analyze aggregated data
        analysis = await client.async_run_agent(
            agent_name="analyzer",
            task=f"Analyze trends in: {scraped_data}",
            model_name="gpt-4",
            temperature=0.5
        )
        
        return analysis

# Run the async function
urls = ["https://example1.com", "https://example2.com"]
result = asyncio.run(scrape_and_analyze(urls))
```

### Real-time Processing with Streaming

```python
from swarms_client import SwarmsClient

client = SwarmsClient(api_key="your-api-key")

# Create streaming swarm
swarm = client.create_swarm(
    name="real-time-processor",
    task="Process incoming data stream",
    agents=[
        {
            "agent_name": "stream-processor",
            "model_name": "gpt-3.5-turbo",
            "role": "worker"
        }
    ],
    stream=True,  # Enable streaming
    service_tier="premium"  # Use premium tier for better performance
)

# Process streaming results
for chunk in swarm['stream']:
    print(f"Received: {chunk}")
    # Process each chunk as it arrives
```

### Error Recovery System

```python
from swarms_client import SwarmsClient, RateLimitError
import time

class ResilientSwarmSystem:
    def __init__(self, api_key):
        self.client = SwarmsClient(
            api_key=api_key,
            max_retries=5,
            retry_delay=2.0
        )
    
    def run_with_fallback(self, task):
        try:
            # Try primary model
            return self.client.run_agent(
                agent_name="primary",
                task=task,
                model_name="gpt-4"
            )
        except RateLimitError:
            # Fallback to secondary model
            print("Rate limit hit, using fallback model")
            return self.client.run_agent(
                agent_name="fallback",
                task=task,
                model_name="gpt-3.5-turbo"
            )
        except Exception as e:
            # Final fallback
            print(f"Error: {e}, using cached response")
            return self.get_cached_response(task)
    
    def get_cached_response(self, task):
        # Implement cache lookup logic
        return {"cached": True, "response": "Cached response"}

# Usage
system = ResilientSwarmSystem(api_key="your-api-key")
result = system.run_with_fallback("Analyze market trends")
```

## Best Practices

### 1. API Key Security

- Never hardcode API keys in your code
- Use environment variables for production
- Rotate keys regularly
- Use different keys for development/production

### 2. Resource Management

```python
# Always use context managers
async with SwarmsClient(api_key="key") as client:
    result = await client.async_run_agent(...)

# Or explicitly close
client = SwarmsClient(api_key="key")
try:
    result = client.run_agent(...)
finally:
    client.close()
```

### 3. Error Handling

```python
# Implement comprehensive error handling
def safe_run_agent(client, **kwargs):
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            return client.run_agent(**kwargs)
        except RateLimitError:
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_attempts - 1:
                raise
```

### 4. Optimize for Performance

```python
# Use batch operations when possible
results = client.run_agent_batch(agents=[...])

# Enable caching for repeated requests
client = SwarmsClient(api_key="key", enable_cache=True)

# Use appropriate concurrency limits
client = SwarmsClient(
    api_key="key",
    max_concurrent_requests=50  # Adjust based on your needs
)
```

### 5. Model Selection

Choose models based on your requirements:
- **GPT-4**: Complex reasoning, analysis, creative tasks
- **GPT-3.5-turbo**: Faster responses, general tasks
- **Claude models**: Extended context, detailed analysis
- **Specialized models**: Domain-specific tasks

### 6. Prompt Engineering

```python
# Be specific with system prompts
agent = AgentSpec(
    agent_name="researcher",
    system_prompt="""You are an expert researcher specializing in:
    1. Academic literature review
    2. Data source verification
    3. Citation formatting (APA style)
    
    Always cite sources and verify facts.""",
    model_name="gpt-4"
)
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Verify API key is correct
   - Check environment variables
   - Ensure key has necessary permissions

2. **Rate Limiting**
   - Implement exponential backoff
   - Use batch operations
   - Consider upgrading service tier

3. **Timeout Errors**
   - Increase timeout setting
   - Break large tasks into smaller chunks
   - Use streaming for long operations

4. **Connection Issues**
   - Check network connectivity
   - Verify firewall settings
   - Use retry logic

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
from loguru import logger

# Enable debug logging
logger.add("swarms_debug.log", level="DEBUG")

# Create client with debug info
client = SwarmsClient(
    api_key="your-key",
    base_url="https://api.swarms.world"
)

# Test connection
try:
    health = client.get_health()
    logger.info(f"Health check: {health}")
except Exception as e:
    logger.error(f"Connection failed: {e}")
```

### Performance Monitoring

```python
import time

class PerformanceMonitor:
    def __init__(self, client):
        self.client = client
        self.metrics = []
    
    def run_with_metrics(self, method, **kwargs):
        start_time = time.time()
        try:
            result = getattr(self.client, method)(**kwargs)
            duration = time.time() - start_time
            self.metrics.append({
                "method": method,
                "duration": duration,
                "success": True
            })
            return result
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.append({
                "method": method,
                "duration": duration,
                "success": False,
                "error": str(e)
            })
            raise
    
    def get_statistics(self):
        successful = [m for m in self.metrics if m["success"]]
        if successful:
            avg_duration = sum(m["duration"] for m in successful) / len(successful)
            return {
                "total_requests": len(self.metrics),
                "successful": len(successful),
                "average_duration": avg_duration,
                "error_rate": (len(self.metrics) - len(successful)) / len(self.metrics)
            }
        return {"error": "No successful requests"}

# Usage
monitor = PerformanceMonitor(client)
result = monitor.run_with_metrics("run_agent", agent_name="test", task="Analyze")
stats = monitor.get_statistics()
print(f"Performance stats: {stats}")
```

## Conclusion

The Swarms API Client provides a robust, production-ready solution for interacting with the Swarms API. With its dual sync/async interface, comprehensive error handling, and performance optimizations, it enables developers to build scalable AI agent systems efficiently. Whether you're creating simple single-agent tasks or complex multi-agent swarms, this client offers the flexibility and reliability needed for production applications.

For the latest updates and additional resources, visit the official documentation at [https://swarms.world](https://swarms.world) and obtain your API keys at [https://swarms.world/platform/api-keys](https://swarms.world/platform/api-keys).