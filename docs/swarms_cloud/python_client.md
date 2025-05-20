# Swarms Cloud API Client Documentation

## Introduction

The Swarms Cloud API client is a production-grade Python package for interacting with the Swarms API. It provides both synchronous and asynchronous interfaces, making it suitable for a wide range of applications from simple scripts to high-performance, scalable services.

Key features include:
- Connection pooling and efficient session management
- Automatic retries with exponential backoff
- Circuit breaker pattern for improved reliability
- In-memory caching for frequently accessed resources
- Comprehensive error handling with detailed exceptions
- Full support for asynchronous operations
- Type checking with Pydantic

This documentation covers all available client methods with detailed descriptions, parameter references, and usage examples.

## Installation

```bash
pip install swarms-client
```

## Authentication

To use the Swarms API, you need an API key. You can obtain your API key from the [Swarms Platform API Keys page](https://swarms.world/platform/api-keys).

## Client Initialization

The `SwarmsClient` is the main entry point for interacting with the Swarms API. It can be initialized with various configuration options to customize its behavior.

```python
from swarms_client import SwarmsClient

# Initialize with default settings
client = SwarmsClient(api_key="your-api-key")

# Or with custom settings
client = SwarmsClient(
    api_key="your-api-key",
    base_url="https://swarms-api-285321057562.us-east1.run.app",
    timeout=60,
    max_retries=3,
    retry_delay=1,
    log_level="INFO",
    pool_connections=100,
    pool_maxsize=100,
    keep_alive_timeout=5,
    max_concurrent_requests=100,
    circuit_breaker_threshold=5,
    circuit_breaker_timeout=60,
    enable_cache=True
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | Environment variable `SWARMS_API_KEY` | API key for authentication |
| `base_url` | `str` | `"https://swarms-api-285321057562.us-east1.run.app"` | Base URL for the API |
| `timeout` | `int` | `60` | Timeout for API requests in seconds |
| `max_retries` | `int` | `3` | Maximum number of retry attempts for failed requests |
| `retry_delay` | `int` | `1` | Initial delay between retries in seconds (uses exponential backoff) |
| `log_level` | `str` | `"INFO"` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `pool_connections` | `int` | `100` | Number of connection pools to cache |
| `pool_maxsize` | `int` | `100` | Maximum number of connections to save in the pool |
| `keep_alive_timeout` | `int` | `5` | Keep-alive timeout for connections in seconds |
| `max_concurrent_requests` | `int` | `100` | Maximum number of concurrent requests |
| `circuit_breaker_threshold` | `int` | `5` | Failure threshold for the circuit breaker |
| `circuit_breaker_timeout` | `int` | `60` | Reset timeout for the circuit breaker in seconds |
| `enable_cache` | `bool` | `True` | Whether to enable in-memory caching |

## Client Methods

### clear_cache

Clears the in-memory cache used for caching API responses.

```python
client.clear_cache()
```

## Agent Resource

The Agent resource provides methods for creating and managing agent completions.

<a name="agent-create"></a>
### create

Creates an agent completion.

```python
response = client.agent.create(
    agent_config={
        "agent_name": "Researcher",
        "description": "Conducts in-depth research on topics",
        "model_name": "gpt-4o",
        "temperature": 0.7
    },
    task="Research the latest advancements in quantum computing and summarize the key findings"
)

print(f"Agent ID: {response.id}")
print(f"Output: {response.outputs}")
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agent_config` | `dict` or `AgentSpec` | Yes | Configuration for the agent |
| `task` | `str` | Yes | The task for the agent to complete |
| `history` | `dict` | No | Optional conversation history |

The `agent_config` parameter can include the following fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `agent_name` | `str` | Required | Name of the agent |
| `description` | `str` | `None` | Description of the agent's purpose |
| `system_prompt` | `str` | `None` | System prompt to guide the agent's behavior |
| `model_name` | `str` | `"gpt-4o-mini"` | Name of the model to use |
| `auto_generate_prompt` | `bool` | `False` | Whether to automatically generate a prompt |
| `max_tokens` | `int` | `8192` | Maximum tokens in the response |
| `temperature` | `float` | `0.5` | Temperature for sampling (0-1) |
| `role` | `str` | `None` | Role of the agent |
| `max_loops` | `int` | `1` | Maximum number of reasoning loops |
| `tools_dictionary` | `List[Dict]` | `None` | Tools available to the agent |

#### Returns

`AgentCompletionResponse` object with the following properties:

- `id`: Unique identifier for the completion
- `success`: Whether the completion was successful
- `name`: Name of the agent
- `description`: Description of the agent
- `temperature`: Temperature used for the completion
- `outputs`: Output from the agent
- `usage`: Token usage information
- `timestamp`: Timestamp of the completion

<a name="agent-create_batch"></a>
### create_batch

Creates multiple agent completions in batch.

```python
responses = client.agent.create_batch([
    {
        "agent_config": {
            "agent_name": "Researcher",
            "model_name": "gpt-4o-mini",
            "temperature": 0.5
        },
        "task": "Summarize the latest quantum computing research"
    },
    {
        "agent_config": {
            "agent_name": "Writer",
            "model_name": "gpt-4o",
            "temperature": 0.7
        },
        "task": "Write a blog post about AI safety"
    }
])

for i, response in enumerate(responses):
    print(f"Agent {i+1} ID: {response.id}")
    print(f"Output: {response.outputs}")
    print("---")
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `completions` | `List[Dict or AgentCompletion]` | Yes | List of agent completion requests |

Each item in the `completions` list should have the same structure as the parameters for the `create` method.

#### Returns

List of `AgentCompletionResponse` objects with the same properties as the return value of the `create` method.

<a name="agent-acreate"></a>
### acreate

Creates an agent completion asynchronously.

```python
import asyncio
from swarms_client import SwarmsClient

async def main():
    async with SwarmsClient(api_key="your-api-key") as client:
        response = await client.agent.acreate(
            agent_config={
                "agent_name": "Researcher",
                "description": "Conducts in-depth research",
                "model_name": "gpt-4o"
            },
            task="Research the impact of quantum computing on cryptography"
        )
        
        print(f"Agent ID: {response.id}")
        print(f"Output: {response.outputs}")

asyncio.run(main())
```

#### Parameters

Same as the `create` method.

#### Returns

Same as the `create` method.

<a name="agent-acreate_batch"></a>
### acreate_batch

Creates multiple agent completions in batch asynchronously.

```python
import asyncio
from swarms_client import SwarmsClient

async def main():
    async with SwarmsClient(api_key="your-api-key") as client:
        responses = await client.agent.acreate_batch([
            {
                "agent_config": {
                    "agent_name": "Researcher",
                    "model_name": "gpt-4o-mini"
                },
                "task": "Summarize the latest quantum computing research"
            },
            {
                "agent_config": {
                    "agent_name": "Writer",
                    "model_name": "gpt-4o"
                },
                "task": "Write a blog post about AI safety"
            }
        ])
        
        for i, response in enumerate(responses):
            print(f"Agent {i+1} ID: {response.id}")
            print(f"Output: {response.outputs}")
            print("---")

asyncio.run(main())
```

#### Parameters

Same as the `create_batch` method.

#### Returns

Same as the `create_batch` method.

## Swarm Resource

The Swarm resource provides methods for creating and managing swarm completions.

<a name="swarm-create"></a>
### create

Creates a swarm completion.

```python
response = client.swarm.create(
    name="Research Swarm",
    description="A swarm for research tasks",
    swarm_type="SequentialWorkflow",
    task="Research quantum computing advances in 2024 and summarize the key findings",
    agents=[
        {
            "agent_name": "Researcher",
            "description": "Conducts in-depth research",
            "model_name": "gpt-4o",
            "temperature": 0.5
        },
        {
            "agent_name": "Critic",
            "description": "Evaluates arguments for flaws",
            "model_name": "gpt-4o-mini",
            "temperature": 0.3
        }
    ],
    max_loops=3,
    return_history=True
)

print(f"Job ID: {response.job_id}")
print(f"Status: {response.status}")
print(f"Output: {response.output}")
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | No | Name of the swarm |
| `description` | `str` | No | Description of the swarm |
| `agents` | `List[Dict or AgentSpec]` | No | List of agent specifications |
| `max_loops` | `int` | No | Maximum number of loops (default: 1) |
| `swarm_type` | `str` | No | Type of swarm (see available types) |
| `task` | `str` | Conditional | The task to complete (required if tasks and messages are not provided) |
| `tasks` | `List[str]` | Conditional | List of tasks for batch processing (required if task and messages are not provided) |
| `messages` | `List[Dict]` | Conditional | List of messages to process (required if task and tasks are not provided) |
| `return_history` | `bool` | No | Whether to return the execution history (default: True) |
| `rules` | `str` | No | Rules for the swarm |
| `schedule` | `Dict` | No | Schedule specification for delayed execution |
| `stream` | `bool` | No | Whether to stream the response (default: False) |
| `service_tier` | `str` | No | Service tier ('standard' or 'flex', default: 'standard') |

#### Returns

`SwarmCompletionResponse` object with the following properties:

- `job_id`: Unique identifier for the job
- `status`: Status of the job
- `swarm_name`: Name of the swarm
- `description`: Description of the swarm
- `swarm_type`: Type of swarm used
- `output`: Output from the swarm
- `number_of_agents`: Number of agents in the swarm
- `service_tier`: Service tier used
- `tasks`: List of tasks processed (if applicable)
- `messages`: List of messages processed (if applicable)

<a name="swarm-create_batch"></a>
### create_batch

Creates multiple swarm completions in batch.

```python
responses = client.swarm.create_batch([
    {
        "name": "Research Swarm",
        "swarm_type": "auto",
        "task": "Research quantum computing advances",
        "agents": [
            {"agent_name": "Researcher", "model_name": "gpt-4o"}
        ]
    },
    {
        "name": "Writing Swarm",
        "swarm_type": "SequentialWorkflow",
        "task": "Write a blog post about AI safety",
        "agents": [
            {"agent_name": "Writer", "model_name": "gpt-4o"},
            {"agent_name": "Editor", "model_name": "gpt-4o-mini"}
        ]
    }
])

for i, response in enumerate(responses):
    print(f"Swarm {i+1} Job ID: {response.job_id}")
    print(f"Status: {response.status}")
    print(f"Output: {response.output}")
    print("---")
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `swarms` | `List[Dict or SwarmSpec]` | Yes | List of swarm specifications |

Each item in the `swarms` list should have the same structure as the parameters for the `create` method.

#### Returns

List of `SwarmCompletionResponse` objects with the same properties as the return value of the `create` method.

<a name="swarm-list_types"></a>
### list_types

Lists available swarm types.

```python
response = client.swarm.list_types()

print(f"Available swarm types:")
for swarm_type in response.swarm_types:
    print(f"- {swarm_type}")
```

#### Returns

`SwarmTypesResponse` object with the following properties:

- `success`: Whether the request was successful
- `swarm_types`: List of available swarm types

<a name="swarm-alist_types"></a>
### alist_types

Lists available swarm types asynchronously.

```python
import asyncio
from swarms_client import SwarmsClient

async def main():
    async with SwarmsClient(api_key="your-api-key") as client:
        response = await client.swarm.alist_types()
        
        print(f"Available swarm types:")
        for swarm_type in response.swarm_types:
            print(f"- {swarm_type}")

asyncio.run(main())
```

#### Returns

Same as the `list_types` method.

<a name="swarm-acreate"></a>
### acreate

Creates a swarm completion asynchronously.

```python
import asyncio
from swarms_client import SwarmsClient

async def main():
    async with SwarmsClient(api_key="your-api-key") as client:
        response = await client.swarm.acreate(
            name="Research Swarm",
            swarm_type="SequentialWorkflow",
            task="Research quantum computing advances in 2024",
            agents=[
                {
                    "agent_name": "Researcher",
                    "description": "Conducts in-depth research",
                    "model_name": "gpt-4o"
                },
                {
                    "agent_name": "Critic",
                    "description": "Evaluates arguments for flaws",
                    "model_name": "gpt-4o-mini"
                }
            ]
        )
        
        print(f"Job ID: {response.job_id}")
        print(f"Status: {response.status}")
        print(f"Output: {response.output}")

asyncio.run(main())
```

#### Parameters

Same as the `create` method.

#### Returns

Same as the `create` method.

<a name="swarm-acreate_batch"></a>
### acreate_batch

Creates multiple swarm completions in batch asynchronously.

```python
import asyncio
from swarms_client import SwarmsClient

async def main():
    async with SwarmsClient(api_key="your-api-key") as client:
        responses = await client.swarm.acreate_batch([
            {
                "name": "Research Swarm",
                "swarm_type": "auto",
                "task": "Research quantum computing",
                "agents": [
                    {"agent_name": "Researcher", "model_name": "gpt-4o"}
                ]
            },
            {
                "name": "Writing Swarm",
                "swarm_type": "SequentialWorkflow",
                "task": "Write a blog post about AI safety",
                "agents": [
                    {"agent_name": "Writer", "model_name": "gpt-4o"}
                ]
            }
        ])
        
        for i, response in enumerate(responses):
            print(f"Swarm {i+1} Job ID: {response.job_id}")
            print(f"Status: {response.status}")
            print(f"Output: {response.output}")
            print("---")

asyncio.run(main())
```

#### Parameters

Same as the `create_batch` method.

#### Returns

Same as the `create_batch` method.

## Models Resource

The Models resource provides methods for retrieving information about available models.

<a name="models-list"></a>
### list

Lists available models.

```python
response = client.models.list()

print(f"Available models:")
for model in response.models:
    print(f"- {model}")
```

#### Returns

`ModelsResponse` object with the following properties:

- `success`: Whether the request was successful
- `models`: List of available model names

<a name="models-alist"></a>
### alist

Lists available models asynchronously.

```python
import asyncio
from swarms_client import SwarmsClient

async def main():
    async with SwarmsClient(api_key="your-api-key") as client:
        response = await client.models.alist()
        
        print(f"Available models:")
        for model in response.models:
            print(f"- {model}")

asyncio.run(main())
```

#### Returns

Same as the `list` method.

## Logs Resource

The Logs resource provides methods for retrieving API request logs.

<a name="logs-list"></a>
### list

Lists API request logs.

```python
response = client.logs.list()

print(f"Found {response.count} logs:")
for log in response.logs:
    print(f"- ID: {log.id}, Created at: {log.created_at}")
    print(f"  Data: {log.data}")
```

#### Returns

`LogsResponse` object with the following properties:

- `status`: Status of the request
- `count`: Number of logs
- `logs`: List of log entries
- `timestamp`: Timestamp of the request

Each log entry is a `LogEntry` object with the following properties:

- `id`: Unique identifier for the log entry
- `api_key`: API key used for the request
- `data`: Request data
- `created_at`: Timestamp when the log entry was created

<a name="logs-alist"></a>
### alist

Lists API request logs asynchronously.

```python
import asyncio
from swarms_client import SwarmsClient

async def main():
    async with SwarmsClient() as client:
        response = await client.logs.alist()
        
        print(f"Found {response.count} logs:")
        for log in response.logs:
            print(f"- ID: {log.id}, Created at: {log.created_at}")
            print(f"  Data: {log.data}")

asyncio.run(main())
```

#### Returns

Same as the `list` method.

## Error Handling

The Swarms API client provides detailed error handling with specific exception types for different error scenarios. All exceptions inherit from the base `SwarmsError` class.

```python
from swarms_client import SwarmsClient, SwarmsError, AuthenticationError, RateLimitError, APIError

try:
    client = SwarmsClient(api_key="invalid-api-key")
    response = client.agent.create(
        agent_config={"agent_name": "Researcher", "model_name": "gpt-4o"},
        task="Research quantum computing"
    )
except AuthenticationError as e:
    print(f"Authentication error: {e}")
except RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
except APIError as e:
    print(f"API error: {e}")
except SwarmsError as e:
    print(f"Swarms error: {e}")
```

### Exception Types

| Exception | Description |
|-----------|-------------|
| `SwarmsError` | Base exception for all Swarms API errors |
| `AuthenticationError` | Raised when there's an issue with authentication |
| `RateLimitError` | Raised when the rate limit is exceeded |
| `APIError` | Raised when the API returns an error |
| `InvalidRequestError` | Raised when the request is invalid |
| `InsufficientCreditsError` | Raised when the user doesn't have enough credits |
| `TimeoutError` | Raised when a request times out |
| `NetworkError` | Raised when there's a network issue |

## Advanced Features

### Connection Pooling

The Swarms API client uses connection pooling to efficiently manage HTTP connections, which can significantly improve performance when making multiple requests.

```python
client = SwarmsClient(
    api_key="your-api-key",
    pool_connections=100,  # Number of connection pools to cache
    pool_maxsize=100,      # Maximum number of connections to save in the pool
    keep_alive_timeout=5   # Keep-alive timeout for connections in seconds
)
```

### Circuit Breaker Pattern

The client implements the circuit breaker pattern to prevent cascading failures when the API is experiencing issues.

```python
client = SwarmsClient(
    api_key="your-api-key",
    circuit_breaker_threshold=5,  # Number of failures before the circuit opens
    circuit_breaker_timeout=60    # Time in seconds before attempting to close the circuit
)
```

### Caching

The client includes in-memory caching for frequently accessed resources to reduce API calls and improve performance.

```python
client = SwarmsClient(
    api_key="your-api-key",
    enable_cache=True  # Enable in-memory caching
)

# Clear the cache manually if needed
client.clear_cache()
```

## Complete Example

Here's a complete example that demonstrates how to use the Swarms API client to create a research swarm and process its output:

```python
import os
from swarms_client import SwarmsClient
from dotenv import load_dotenv

# Load API key from environment
load_dotenv()
api_key = os.getenv("SWARMS_API_KEY")

# Initialize client
client = SwarmsClient(api_key=api_key)

# Create a research swarm
try:
    # Define the agents
    researcher = {
        "agent_name": "Researcher",
        "description": "Conducts thorough research on specified topics",
        "model_name": "gpt-4o",
        "temperature": 0.5,
        "system_prompt": "You are a diligent researcher focused on finding accurate and comprehensive information."
    }
    
    analyst = {
        "agent_name": "Analyst",
        "description": "Analyzes research findings and identifies key insights",
        "model_name": "gpt-4o",
        "temperature": 0.3,
        "system_prompt": "You are an insightful analyst who can identify patterns and extract meaningful insights from research data."
    }
    
    summarizer = {
        "agent_name": "Summarizer",
        "description": "Creates concise summaries of complex information",
        "model_name": "gpt-4o-mini",
        "temperature": 0.4,
        "system_prompt": "You specialize in distilling complex information into clear, concise summaries."
    }
    
    # Create the swarm
    response = client.swarm.create(
        name="Quantum Computing Research Swarm",
        description="A swarm for researching and analyzing quantum computing advancements",
        swarm_type="SequentialWorkflow",
        task="Research the latest advancements in quantum computing in 2024, analyze their potential impact on cryptography and data security, and provide a concise summary of the findings.",
        agents=[researcher, analyst, summarizer],
        max_loops=2,
        return_history=True
    )
    
    # Process the response
    print(f"Job ID: {response.job_id}")
    print(f"Status: {response.status}")
    print(f"Number of agents: {response.number_of_agents}")
    print(f"Swarm type: {response.swarm_type}")
    
    # Print the output
    if "final_output" in response.output:
        print("\nFinal Output:")
        print(response.output["final_output"])
    else:
        print("\nOutput:")
        print(response.output)
    
    # Access agent-specific outputs if available
    if "agent_outputs" in response.output:
        print("\nAgent Outputs:")
        for agent, output in response.output["agent_outputs"].items():
            print(f"\n{agent}:")
            print(output)

except Exception as e:
    print(f"Error: {e}")
```

This example creates a sequential workflow swarm with three agents to research quantum computing, analyze the findings, and create a summary of the results.
