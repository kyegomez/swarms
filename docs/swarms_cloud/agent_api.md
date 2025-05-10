# Agent API

The Swarms.ai Agent API provides powerful endpoints for running individual AI agents and batch agent operations. This documentation explains how to use these endpoints for effective agent-based task execution.

## Getting Started

To use the Agent API, you'll need a Swarms.ai API key:

1. Go to [https://swarms.world/platform/api-keys](https://swarms.world/platform/api-keys)
2. Generate a new API key
3. Store your API key securely - it won't be shown again

```python
import os
import requests
from dotenv import load_dotenv

# Load API key from environment
load_dotenv()
API_KEY = os.getenv("SWARMS_API_KEY")
BASE_URL = "https://api.swarms.world"

# Configure headers with your API key
headers = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}
```

## Individual Agent API

The Individual Agent API allows you to run a single agent with a specific configuration and task.

### Agent Configuration (`AgentSpec`)

The `AgentSpec` class defines the configuration for an individual agent.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_name` | string | Required | Unique name identifying the agent and its functionality |
| `description` | string | None | Detailed explanation of the agent's purpose and capabilities |
| `system_prompt` | string | None | Initial instructions guiding the agent's behavior and responses |
| `model_name` | string | "gpt-4o-mini" | The AI model used by the agent (e.g., gpt-4o, gpt-4o-mini, openai/o3-mini) |
| `auto_generate_prompt` | boolean | false | Whether the agent should automatically create prompts based on task requirements |
| `max_tokens` | integer | 8192 | Maximum number of tokens the agent can generate in its responses |
| `temperature` | float | 0.5 | Controls output randomness (lower values = more deterministic responses) |
| `role` | string | "worker" | The agent's role within a swarm, influencing its behavior and interactions |
| `max_loops` | integer | 1 | Maximum number of times the agent can repeat its task for iterative processing |
| `tools_dictionary` | array | None | Dictionary of tools the agent can use to complete its task |

### Agent Completion

The `AgentCompletion` class combines an agent configuration with a specific task.

| Parameter | Type | Description |
|-----------|------|-------------|
| `agent_config` | AgentSpec | Configuration of the agent to be completed |
| `task` | string | The task to be completed by the agent |

### Single Agent Endpoint

**Endpoint:** `POST /v1/agent/completions`

Run a single agent with a specific configuration and task.

#### Request

```python
def run_single_agent(agent_config, task):
    """
    Run a single agent with the AgentCompletion format.
    
    Args:
        agent_config: Dictionary containing agent configuration
        task: String describing the task for the agent
        
    Returns:
        Dictionary containing the agent's response
    """
    payload = {
        "agent_config": agent_config,
        "task": task
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/v1/agent/completions", 
            headers=headers, 
            json=payload
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None
```

#### Example Usage

```python
agent_config = {
    "agent_name": "Research Analyst",
    "description": "An expert in analyzing and synthesizing research data",
    "system_prompt": (
        "You are a Research Analyst with expertise in data analysis and synthesis. "
        "Your role is to analyze provided information, identify key insights, "
        "and present findings in a clear, structured format. "
        "Focus on accuracy, clarity, and actionable recommendations."
    ),
    "model_name": "gpt-4o",
    "role": "worker",
    "max_loops": 2,
    "max_tokens": 8192,
    "temperature": 0.5,
    "auto_generate_prompt": False,
}

task = "Analyze the impact of artificial intelligence on healthcare delivery and provide a comprehensive report with key findings and recommendations."

result = run_single_agent(agent_config, task)
print(result)
```

#### Response Structure

```json
{
  "id": "agent-6a8b9c0d1e2f3g4h5i6j7k8l9m0n",
  "success": true,
  "name": "Research Analyst",
  "description": "An expert in analyzing and synthesizing research data",
  "temperature": 0.5,
  "outputs": {
    "content": "# Impact of Artificial Intelligence on Healthcare Delivery\n\n## Executive Summary\n...",
    "role": "assistant"
  },
  "usage": {
    "input_tokens": 1250,
    "output_tokens": 3822,
    "total_tokens": 5072
  },
  "timestamp": "2025-05-10T18:35:29.421Z"
}
```

## Batch Agent API

The Batch Agent API allows you to run multiple agents in parallel, each with different configurations and tasks.

### Batch Agent Endpoint

**Endpoint:** `POST /v1/agent/batch/completions`

Run multiple agents with different configurations and tasks in a single API call.

#### Request

```python
def run_batch_agents(agent_completions):
    """
    Run multiple agents in batch.
    
    Args:
        agent_completions: List of dictionaries, each containing agent_config and task
        
    Returns:
        List of agent responses
    """
    try:
        response = requests.post(
            f"{BASE_URL}/v1/agent/batch/completions",
            headers=headers,
            json=agent_completions
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making batch request: {e}")
        return None
```

#### Example Usage

```python
batch_completions = [
    {
        "agent_config": {
            "agent_name": "Research Analyst",
            "description": "An expert in analyzing research data",
            "system_prompt": "You are a Research Analyst...",
            "model_name": "gpt-4o",
            "max_loops": 2
        },
        "task": "Analyze the impact of AI on healthcare delivery."
    },
    {
        "agent_config": {
            "agent_name": "Market Analyst",
            "description": "An expert in market analysis",
            "system_prompt": "You are a Market Analyst...",
            "model_name": "gpt-4o",
            "max_loops": 1
        },
        "task": "Analyze the AI startup landscape in 2025."
    }
]

batch_results = run_batch_agents(batch_completions)
print(batch_results)
```

#### Response Structure

```json
[
  {
    "id": "agent-1a2b3c4d5e6f7g8h9i0j",
    "success": true,
    "name": "Research Analyst",
    "description": "An expert in analyzing research data",
    "temperature": 0.5,
    "outputs": {
      "content": "# Impact of AI on Healthcare Delivery\n...",
      "role": "assistant"
    },
    "usage": {
      "input_tokens": 1250,
      "output_tokens": 3822,
      "total_tokens": 5072
    },
    "timestamp": "2025-05-10T18:35:29.421Z"
  },
  {
    "id": "agent-9i8h7g6f5e4d3c2b1a0",
    "success": true,
    "name": "Market Analyst",
    "description": "An expert in market analysis",
    "temperature": 0.5,
    "outputs": {
      "content": "# AI Startup Landscape 2025\n...",
      "role": "assistant"
    },
    "usage": {
      "input_tokens": 980,
      "output_tokens": 4120,
      "total_tokens": 5100
    },
    "timestamp": "2025-05-10T18:35:31.842Z"
  }
]
```

## Error Handling

The API uses standard HTTP status codes to indicate success or failure:

| Status Code | Meaning |
|-------------|---------|
| 200 | Success |
| 400 | Bad Request - Check your request parameters |
| 401 | Unauthorized - Invalid or missing API key |
| 403 | Forbidden - Insufficient permissions |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Server Error - Something went wrong on the server |

When an error occurs, the response body will contain additional information:

```json
{
  "detail": "Error message explaining what went wrong"
}
```

### Common Errors and Solutions

| Error | Possible Solution |
|-------|-------------------|
| "Invalid API Key" | Verify your API key is correct and properly included in the request headers |
| "Rate limit exceeded" | Reduce the number of requests or contact support to increase your rate limit |
| "Invalid agent configuration" | Check your agent_config parameters for any missing or invalid values |
| "Failed to create agent" | Ensure your system_prompt and model_name are valid |
| "Insufficient credits" | Add credits to your account at https://swarms.world/platform/account |

## Advanced Usage

### Setting Dynamic Temperature

The agent can dynamically adjust its temperature for optimal outputs:

```python
agent_config = {
    # Other config options...
    "temperature": 0.7,
    "dynamic_temperature_enabled": True
}
```

### Using Agent Tools

Agents can utilize various tools to enhance their capabilities:

```python
agent_config = {
    # Other config options...
    "tools_dictionary": [
        {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "query": "string"
            }
        },
        {
            "name": "calculator",
            "description": "Perform mathematical calculations",
            "parameters": {
                "expression": "string"
            }
        }
    ]
}
```

## Best Practices

!!! tip "API Key Security"
    Store API keys in environment variables or secure vaults, never in code repositories.
    ```python
    # DON'T do this
    api_key = "sk-123456789abcdef"
    
    # DO this instead
    import os
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("SWARMS_API_KEY")
    ```

!!! info "Agent Naming Conventions"
    Use a consistent naming pattern for your agents to make your code more maintainable.
    ```python
    # Good naming convention
    agent_configs = {
        "market_analyst": {...},
        "research_specialist": {...},
        "code_reviewer": {...}
    }
    ```

!!! success "Crafting Effective System Prompts"
    A well-crafted system prompt acts as your agent's personality and instruction set.
    
    === "Basic Prompt"
        ```
        You are a research analyst. Analyze the data and provide insights.
        ```
    
    === "Enhanced Prompt"
        ```
        You are a Research Analyst with 15+ years of experience in biotech market analysis.
        
        Your task is to:
        1. Analyze the provided market data methodically
        2. Identify key trends and emerging patterns
        3. Highlight potential investment opportunities
        4. Assess risks and regulatory considerations
        5. Provide actionable recommendations supported by the data
        
        Format your response as a professional report with clear sections,
        focusing on data-driven insights rather than generalities.
        ```

!!! warning "Token Management"
    Manage your token usage carefully to control costs.
    
    - Higher token limits provide more complete responses but increase costs
    - Consider using different models based on task complexity
    - For gpt-4o models, typical settings:
        - Simple tasks: 2048 tokens (lower cost)
        - Medium complexity: 4096 tokens (balanced)
        - Complex analysis: 8192+ tokens (higher cost, more detail)

!!! danger "Error Handling"
    Implement comprehensive error handling to make your application resilient.
    
    ```python
    try:
        response = requests.post(
            f"{BASE_URL}/v1/agent/completions",
            headers=headers,
            json=payload,
            timeout=30  # Add timeout to prevent hanging requests
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            # Implement exponential backoff for rate limiting
            retry_after = int(e.response.headers.get('Retry-After', 5))
            time.sleep(retry_after)
            return run_agent(payload)  # Retry the request
        elif e.response.status_code == 401:
            logger.error("Authentication failed. Check your API key.")
        else:
            logger.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
        return {"error": e.response.text}
    except requests.exceptions.Timeout:
        logger.error("Request timed out. The server might be busy.")
        return {"error": "Request timed out"}
    except requests.exceptions.RequestException as e:
        logger.error(f"Request Error: {e}")
        return {"error": str(e)}
    ```

!!! example "Implementing Caching"
    Cache identical requests to improve performance and reduce costs.
    
    ```python
    import hashlib
    import json
    from functools import lru_cache
    
    def generate_cache_key(agent_config, task):
        """Generate a unique cache key for an agent request."""
        cache_data = json.dumps({"agent_config": agent_config, "task": task}, sort_keys=True)
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    @lru_cache(maxsize=100)
    def cached_agent_run(cache_key, agent_config, task):
        """Run agent with caching based on config and task."""
        # Convert agent_config back to a dictionary if it's a string representation
        if isinstance(agent_config, str):
            agent_config = json.loads(agent_config)
            
        payload = {
            "agent_config": agent_config,
            "task": task
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/v1/agent/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def run_agent_with_cache(agent_config, task):
        """Wrapper function to run agent with caching."""
        # Generate a cache key
        cache_key = generate_cache_key(agent_config, task)
        
        # Convert agent_config to a hashable type for lru_cache
        hashable_config = json.dumps(agent_config, sort_keys=True)
        
        # Call the cached function
        return cached_agent_run(cache_key, hashable_config, task)
    ```

!!! abstract "Usage & Cost Monitoring"
    Set up a monitoring system to track your API usage and costs.
    
    ```python
    def log_api_usage(api_call_type, tokens_used, cost_estimate):
        """Log API usage for monitoring."""
        with open("api_usage_log.csv", "a") as f:
            timestamp = datetime.now().isoformat()
            f.write(f"{timestamp},{api_call_type},{tokens_used},{cost_estimate}\n")
    
    def estimate_cost(tokens):
        """Estimate cost based on token usage."""
        # Example pricing: $0.002 per 1K tokens (adjust according to current pricing)
        return (tokens / 1000) * 0.002
    
    def run_agent_with_logging(agent_config, task):
        """Run agent and log usage."""
        result = run_single_agent(agent_config, task)
        
        if "usage" in result:
            total_tokens = result["usage"]["total_tokens"]
            cost = estimate_cost(total_tokens)
            log_api_usage("single_agent", total_tokens, cost)
            
        return result
    ```

## FAQ

??? question "What's the difference between Single Agent and Batch Agent APIs?"
    The Single Agent API (`/v1/agent/completions`) runs one agent with one task, while the Batch Agent API (`/v1/agent/batch/completions`) allows running multiple agents with different configurations and tasks in parallel. Use Batch Agent when you need to process multiple independent tasks efficiently.

??? question "How do I choose the right model for my agent?"
    Model selection depends on your task complexity, performance requirements, and budget:
    
    | Model | Best For | Characteristics |
    |-------|----------|-----------------|
    | gpt-4o | Complex analysis, creative tasks | Highest quality, most expensive |
    | gpt-4o-mini | General purpose tasks | Good balance of quality and cost |
    | openai/o3-mini | Simple, factual tasks | Fast, economical |
    
    For exploratory work, start with gpt-4o-mini and adjust based on results.

??? question "What should I include in my system prompt?"
    A good system prompt should include:
    
    1. **Role definition**: Who the agent is and their expertise
    2. **Task instructions**: Specific, clear directions on what to do
    3. **Output format**: How results should be structured
    4. **Constraints**: Any limitations or requirements
    5. **Examples**: Sample inputs and outputs when helpful
    
    Keep prompts focused and avoid contradictory instructions.

??? question "How can I optimize costs when using the Agent API?"
    Cost optimization strategies include:
    
    - Use the appropriate model for your task complexity
    - Set reasonable token limits based on expected output length
    - Implement caching for repeated or similar requests
    - Batch related requests together
    - Use `max_loops: 1` unless you specifically need iterative refinement
    - Monitor usage patterns and adjust configurations accordingly

??? question "What's the maximum number of agents I can run in a batch?"
    While there's no hard limit specified, we recommend keeping batch sizes under 20 agents for optimal performance. For very large batches, consider splitting them into multiple calls or contacting support for guidance on handling high-volume processing.

??? question "How do I handle rate limiting?"
    Implement exponential backoff in your error handling:
    
    ```python
    import time
    
    def run_with_backoff(func, max_retries=5, initial_delay=1):
        """Run a function with exponential backoff retry logic."""
        retries = 0
        delay = initial_delay
        
        while retries < max_retries:
            try:
                return func()
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Too Many Requests
                    retry_after = int(e.response.headers.get('Retry-After', delay))
                    print(f"Rate limited. Retrying after {retry_after} seconds...")
                    time.sleep(retry_after)
                    retries += 1
                    delay *= 2  # Exponential backoff
                else:
                    raise
            except Exception as e:
                raise
                
        raise Exception(f"Failed after {max_retries} retries")
    ```

??? question "Can I use tools with my agents?"
    Yes, you can enable tools through the `tools_dictionary` parameter in your agent configuration. This allows agents to access external functionality like web searches, calculations, or custom tools.
    
    ```python
    agent_config = {
        # Other configuration...
        "tools_dictionary": [
            {
                "name": "web_search",
                "description": "Search the web for current information",
                "parameters": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                }
            }
        ]
    }
    ```

??? question "How do I debug agent performance issues?"
    Debugging steps for agent performance issues:
    
    1. **Check system prompts**: Ensure they're clear and not overly restrictive
    2. **Review model selection**: Try a more capable model if output quality is poor
    3. **Adjust token limits**: Increase max_tokens if outputs are getting truncated
    4. **Examine temperature**: Lower for more deterministic outputs, higher for creativity
    5. **Test with simpler tasks**: Isolate whether the issue is with the task complexity
    6. **Enable verbose logging**: Add detailed logging to track request/response cycles
    7. **Contact support**: For persistent issues, reach out with example payloads and responses

??? question "What's the pricing model for the Agent API?"
    The Agent API uses a token-based pricing model:
    
    1. **Input tokens**: Text sent to the API (task, system prompts)
    2. **Output tokens**: Text generated by the agent
    
    Pricing varies by model and is calculated per 1,000 tokens. Check the [pricing page](https://swarms.world/platform/pricing) for current rates.
    
    The API also offers a "flex" tier for lower-priority, cost-effective processing.

## Further Resources

[:material-file-document: Swarms.ai Documentation](https://docs.swarms.world){ .md-button }
[:material-application: Swarms.ai Platform](https://swarms.world/platform){ .md-button }
[:material-key: API Key Management](https://swarms.world/platform/api-keys){ .md-button }
[:material-forum: Swarms.ai Community](https://discord.gg/swarms){ .md-button }