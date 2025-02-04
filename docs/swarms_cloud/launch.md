# Swarms Cloud API Client Documentation

## Overview
The Swarms Cloud API Client is a production-grade Python library for interacting with the Swarms Cloud Agent API. It provides a comprehensive interface for managing, executing, and monitoring cloud-based agents.

## Installation
```bash
pip install swarms-cloud
```

## Quick Start
```python
from swarms_cloud import SwarmCloudAPI, AgentCreate

# Initialize the client
client = SwarmCloudAPI(
    base_url="https://swarmcloud-285321057562.us-central1.run.app",
    api_key="your_api_key_here"
)

# Create an agent
agent_data = AgentCreate(
    name="TranslateAgent",
    description="Translates text between languages",
    code="""
    def main(request, store):
        text = request.payload.get('text', '')
        return f'Translated: {text}'
    """,
    requirements="requests==2.25.1",
    envs="DEBUG=True"
)

new_agent = client.create_agent(agent_data)
print(f"Created agent with ID: {new_agent.id}")
```

## Client Configuration

### Constructor Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|----------|-------------|
| base_url | str | No | https://swarmcloud-285321057562.us-central1.run.app | The base URL of the SwarmCloud API |
| api_key | str | Yes | None | Your SwarmCloud API key |
| timeout | float | No | 10.0 | Request timeout in seconds |

## Data Models

### AgentCreate
Model for creating new agents.

| Field | Type | Required | Default | Description |
|-------|------|----------|----------|-------------|
| name | str | Yes | - | Name of the agent |
| description | str | No | None | Description of the agent's purpose |
| code | str | Yes | - | Python code that defines the agent's behavior |
| requirements | str | No | None | Python package requirements (pip format) |
| envs | str | No | None | Environment variables for the agent |
| autoscaling | bool | No | False | Enable/disable concurrent execution scaling |

### AgentUpdate
Model for updating existing agents.

| Field | Type | Required | Default | Description |
|-------|------|----------|----------|-------------|
| name | str | No | None | Updated name of the agent |
| description | str | No | None | Updated description |
| code | str | No | None | Updated Python code |
| requirements | str | No | None | Updated package requirements |
| autoscaling | bool | No | None | Updated autoscaling setting |

## API Methods

### List Agents
Retrieve all available agents.

```python
agents = client.list_agents()
for agent in agents:
    print(f"Agent: {agent.name} (ID: {agent.id})")
```

**Returns**: List[AgentOut]

### Create Agent
Create a new agent with the specified configuration.

```python
agent_data = AgentCreate(
    name="DataProcessor",
    description="Processes incoming data streams",
    code="""
    def main(request, store):
        data = request.payload.get('data', [])
        return {'processed': len(data)}
    """,
    requirements="pandas==1.4.0\nnumpy==1.21.0",
    envs="PROCESSING_MODE=fast",
    autoscaling=True
)

new_agent = client.create_agent(agent_data)
```

**Returns**: AgentOut

### Get Agent
Retrieve details of a specific agent.

```python
agent = client.get_agent("agent_id_here")
print(f"Agent details: {agent}")
```

**Parameters**:
- agent_id (str): The unique identifier of the agent

**Returns**: AgentOut

### Update Agent
Update an existing agent's configuration.

```python
update_data = AgentUpdate(
    name="UpdatedProcessor",
    description="Enhanced data processing capabilities",
    code="def main(request, store):\n    return {'status': 'updated'}"
)

updated_agent = client.update_agent("agent_id_here", update_data)
```

**Parameters**:
- agent_id (str): The unique identifier of the agent
- update (AgentUpdate): The update data

**Returns**: AgentOut

### Execute Agent
Manually execute an agent with optional payload data.

```python
# Execute with payload
result = client.execute_agent(
    "agent_id_here",
    payload={"text": "Hello, World!"}
)

# Execute without payload
result = client.execute_agent("agent_id_here")
```

**Parameters**:
- agent_id (str): The unique identifier of the agent
- payload (Optional[Dict[str, Any]]): Execution payload data

**Returns**: Dict[str, Any]

### Get Agent History
Retrieve the execution history and logs for an agent.

```python
history = client.get_agent_history("agent_id_here")
for execution in history.executions:
    print(f"[{execution.timestamp}] {execution.log}")
```

**Parameters**:
- agent_id (str): The unique identifier of the agent

**Returns**: AgentExecutionHistory

### Batch Execute Agents
Execute multiple agents simultaneously with the same payload.

```python
# Get list of agents
agents = client.list_agents()

# Execute batch with payload
results = client.batch_execute_agents(
    agents=agents[:3],  # Execute first three agents
    payload={"data": "test"}
)

print(f"Batch execution results: {results}")
```

**Parameters**:
- agents (List[AgentOut]): List of agents to execute
- payload (Optional[Dict[str, Any]]): Shared execution payload

**Returns**: List[Any]

### Health Check
Check the API's health status.

```python
status = client.health()
print(f"API Status: {status}")
```

**Returns**: Dict[str, Any]

## Error Handling
The client uses exception handling to manage various error scenarios:

```python
from swarms_cloud import SwarmCloudAPI
import httpx

try:
    client = SwarmCloudAPI(api_key="your_api_key_here")
    agents = client.list_agents()
except httpx.HTTPError as http_err:
    print(f"HTTP error occurred: {http_err}")
except Exception as err:
    print(f"An unexpected error occurred: {err}")
finally:
    client.close()
```

## Context Manager Support
The client can be used with Python's context manager:

```python
with SwarmCloudAPI(api_key="your_api_key_here") as client:
    status = client.health()
    print(f"API Status: {status}")
    # Client automatically closes after the with block
```

## Best Practices

1. Always close the client when finished:
```python
client = SwarmCloudAPI(api_key="your_api_key_here")
try:
    # Your code here
finally:
    client.close()
```

2. Use context managers for automatic cleanup:
```python
with SwarmCloudAPI(api_key="your_api_key_here") as client:
    # Your code here
```

3. Handle errors appropriately:
```python
try:
    result = client.execute_agent("agent_id", payload={"data": "test"})
except httpx.HTTPError as e:
    logger.error(f"HTTP error: {e}")
    # Handle error appropriately
```

4. Set appropriate timeouts for your use case:
```python
client = SwarmCloudAPI(
    api_key="your_api_key_here",
    timeout=30.0  # Longer timeout for complex operations
)
```

## Complete Example
Here's a complete example showcasing various features of the client:

```python
from swarms_cloud import SwarmCloudAPI, AgentCreate, AgentUpdate
import httpx

def main():
    with SwarmCloudAPI(api_key="your_api_key_here") as client:
        # Create an agent
        agent_data = AgentCreate(
            name="DataAnalyzer",
            description="Analyzes incoming data streams",
            code="""
            def main(request, store):
                data = request.payload.get('data', [])
                return {
                    'count': len(data),
                    'summary': 'Data processed successfully'
                }
            """,
            requirements="pandas==1.4.0",
            autoscaling=True
        )
        
        try:
            # Create the agent
            new_agent = client.create_agent(agent_data)
            print(f"Created agent: {new_agent.name} (ID: {new_agent.id})")
            
            # Execute the agent
            result = client.execute_agent(
                new_agent.id,
                payload={"data": [1, 2, 3, 4, 5]}
            )
            print(f"Execution result: {result}")
            
            # Update the agent
            update_data = AgentUpdate(
                description="Enhanced data analysis capabilities"
            )
            updated_agent = client.update_agent(new_agent.id, update_data)
            print(f"Updated agent: {updated_agent.name}")
            
            # Get execution history
            history = client.get_agent_history(new_agent.id)
            print(f"Execution history: {history}")
            
        except httpx.HTTPError as e:
            print(f"HTTP error occurred: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
```

## Logging
The client uses the `loguru` library for logging. You can configure the logging level and format:

```python
from loguru import logger

# Configure logging
logger.add("swarmcloud.log", rotation="500 MB")

client = SwarmCloudAPI(api_key="your_api_key_here")
```

## Performance Considerations

1. **Connection Reuse**: The client reuses HTTP connections by default, improving performance for multiple requests.

2. **Timeout Configuration**: Set appropriate timeouts based on your use case:
```python
client = SwarmCloudAPI(
    api_key="your_api_key_here",
    timeout=5.0  # Shorter timeout for time-sensitive operations
)
```

3. **Batch Operations**: Use batch_execute_agents for multiple agent executions:
```python
results = client.batch_execute_agents(
    agents=agents,
    payload=shared_payload
)
```

## Rate Limiting
The client respects API rate limits but does not implement retry logic. Implement your own retry mechanism if needed:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def execute_with_retry(client, agent_id, payload):
    return client.execute_agent(agent_id, payload)
```

## Thread Safety
The client is not thread-safe by default. For concurrent usage, create separate client instances for each thread or implement appropriate synchronization mechanisms.