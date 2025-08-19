# Swarms API Documentation

*Enterprise-Grade Agent Swarm Management API*

**Base URL**: `https://api.swarms.world` or `https://swarms-api-285321057562.us-east1.run.app`

**API Key Management**: [https://swarms.world/platform/api-keys](https://swarms.world/platform/api-keys)  

## Overview

The Swarms API provides a robust, scalable infrastructure for deploying and managing intelligent agent swarms in the cloud. This enterprise-grade API enables organizations to create, execute, and orchestrate sophisticated AI agent workflows without managing the underlying infrastructure.

Key capabilities include:

- **Intelligent Swarm Management**: Create and execute swarms of specialized AI agents that collaborate to solve complex tasks

- **Automatic Agent Generation**: Dynamically create optimized agents based on task requirements

- **Multiple Swarm Architectures**: Choose from various swarm patterns to match your specific workflow needs

- **Comprehensive Logging**: Track and analyze all API interactions

- **Cost Management**: Predictable, transparent pricing with optimized resource utilization

- **Enterprise Security**: Full API key authentication and management

Swarms API is designed for production use cases requiring sophisticated AI orchestration, with applications in finance, healthcare, legal, research, and other domains where complex reasoning and multi-agent collaboration are needed.

## Authentication

All API requests require a valid API key, which must be included in the header of each request:

```
x-api-key: your_api_key_here
```

API keys can be obtained and managed at [https://swarms.world/platform/api-keys](https://swarms.world/platform/api-keys).

## API Reference

### Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Simple health check endpoint |
| `/v1/swarm/completions` | POST | Run a swarm with specified configuration |
| `/v1/swarm/batch/completions` | POST | Run multiple swarms in batch mode |
| `/v1/swarm/logs` | GET | Retrieve API request logs |
| `/v1/swarms/available` | GET | Get all available swarms as a list of strings |
| `/v1/models/available` | GET | Get all available models as a list of strings |
| `/v1/agent/completions` | POST | Run a single agent with specified configuration |
| `/v1/agent/batch/completions` | POST | Run a batch of individual agent completions|



### SwarmType Reference

The `swarm_type` parameter defines the architecture and collaboration pattern of the agent swarm:

| SwarmType | Description |
|-----------|-------------|
| `AgentRearrange` | Dynamically reorganizes the workflow between agents based on task requirements |
| `MixtureOfAgents` | Combines multiple agent types to tackle diverse aspects of a problem |
| `SpreadSheetSwarm` | Specialized for spreadsheet data analysis and manipulation |
| `SequentialWorkflow` | Agents work in a predefined sequence, each handling specific subtasks |
| `ConcurrentWorkflow` | Multiple agents work simultaneously on different aspects of the task |
| `GroupChat` | Agents collaborate in a discussion format to solve problems |
| `MultiAgentRouter` | Routes subtasks to specialized agents based on their capabilities |
| `AutoSwarmBuilder` | Automatically designs and builds an optimal swarm based on the task |
| `HiearchicalSwarm` | Organizes agents in a hierarchical structure with managers and workers |
| `MajorityVoting` | Uses a consensus mechanism where multiple agents vote on the best solution |
| `auto` | Automatically selects the most appropriate swarm type for the given task |



## Data Models

### SwarmSpec

The `SwarmSpec` model defines the configuration of a swarm.

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| name | string | Identifier for the swarm | No |
| description | string | Description of the swarm's purpose | No |
| agents | Array<AgentSpec> | List of agent specifications | No |
| max_loops | integer | Maximum number of execution loops | No |
| swarm_type | SwarmType | Architecture of the swarm | No |
| rearrange_flow | string | Instructions for rearranging task flow | No |
| task | string | The main task for the swarm to accomplish | Yes |
| img | string | Optional image URL for the swarm | No |
| return_history | boolean | Whether to return execution history | No |
| rules | string | Guidelines for swarm behavior | No |
| service_tier | string | Service tier for processing ("standard" or "flex") | No |

### AgentSpec

The `AgentSpec` model defines the configuration of an individual agent.

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| agent_name | string | Unique name for the agent | Yes* |
| description | string | Description of the agent's purpose | No |
| system_prompt | string | Instructions for the agent | No |
| model_name | string | AI model to use (e.g., "gpt-4o") | Yes* |
| auto_generate_prompt | boolean | Whether to auto-generate prompts | No |
| max_tokens | integer | Maximum tokens in response | No |
| temperature | float | Randomness of responses (0-1) | No |
| role | string | Agent's role in the swarm | No |
| max_loops | integer | Maximum iterations for this agent | No |

*Required if agents are manually specified; not required if using auto-generated agents


### Endpoint Details

#### Health Check

Check if the API service is available and functioning correctly.

**Endpoint**: `/health`  
**Method**: GET  
**Rate Limit**: 100 requests per 60 seconds

=== "Shell (curl)"
    ```bash
    curl -X GET "https://api.swarms.world/health" \
         -H "x-api-key: your_api_key_here"
    ```

=== "Python (requests)"
    ```python
    import requests

    API_BASE_URL = "https://api.swarms.world"
    API_KEY = "your_api_key_here"
    
    headers = {
        "x-api-key": API_KEY
    }
    
    response = requests.get(f"{API_BASE_URL}/health", headers=headers)
    
    if response.status_code == 200:
        print("API is healthy:", response.json())
    else:
        print(f"Error: {response.status_code}")
    ```

=== "TypeScript (fetch)"
    ```typescript
    const API_BASE_URL = "https://api.swarms.world";
    const API_KEY = "your_api_key_here";

    async function checkHealth(): Promise<void> {
        try {
            const response = await fetch(`${API_BASE_URL}/health`, {
                method: 'GET',
                headers: {
                    'x-api-key': API_KEY
                }
            });

            if (response.ok) {
                const data = await response.json();
                console.log("API is healthy:", data);
            } else {
                console.error(`Error: ${response.status}`);
            }
        } catch (error) {
            console.error("Request failed:", error);
        }
    }

    checkHealth();
    ```

**Example Response**:
```json
{
  "status": "ok"
}
```

#### Run Swarm

Run a swarm with the specified configuration to complete a task.

**Endpoint**: `/v1/swarm/completions`  
**Method**: POST  
**Rate Limit**: 100 requests per 60 seconds

**Request Parameters**:

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| name | string | Identifier for the swarm | No |
| description | string | Description of the swarm's purpose | No |
| agents | Array<AgentSpec> | List of agent specifications | No |
| max_loops | integer | Maximum number of execution loops | No |
| swarm_type | SwarmType | Architecture of the swarm | No |
| rearrange_flow | string | Instructions for rearranging task flow | No |
| task | string | The main task for the swarm to accomplish | Yes |
| img | string | Optional image URL for the swarm | No |
| return_history | boolean | Whether to return execution history | No |
| rules | string | Guidelines for swarm behavior | No |

=== "Shell (curl)"
    ```bash
    curl -X POST "https://api.swarms.world/v1/swarm/completions" \
      -H "x-api-key: $SWARMS_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
        "name": "Financial Analysis Swarm",
        "description": "Market analysis swarm",
        "agents": [
          {
            "agent_name": "Market Analyst",
            "description": "Analyzes market trends",
            "system_prompt": "You are a financial analyst expert.",
            "model_name": "openai/gpt-4o",
            "role": "worker",
            "max_loops": 1,
            "max_tokens": 8192,
            "temperature": 0.5,
            "auto_generate_prompt": false
          },
          {
            "agent_name": "Economic Forecaster",
            "description": "Predicts economic trends",
            "system_prompt": "You are an expert in economic forecasting.",
            "model_name": "gpt-4o",
            "role": "worker",
            "max_loops": 1,
            "max_tokens": 8192,
            "temperature": 0.5,
            "auto_generate_prompt": false
          }
        ],
        "max_loops": 1,
        "swarm_type": "ConcurrentWorkflow",
        "task": "What are the best etfs and index funds for ai and tech?",
        "output_type": "dict"
      }'
    ```

=== "Python (requests)"
    ```python
    import requests
    import json

    API_BASE_URL = "https://api.swarms.world"
    API_KEY = "your_api_key_here"
    
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    
    swarm_config = {
        "name": "Financial Analysis Swarm",
        "description": "Market analysis swarm",
        "agents": [
            {
                "agent_name": "Market Analyst",
                "description": "Analyzes market trends",
                "system_prompt": "You are a financial analyst expert.",
                "model_name": "openai/gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
                "temperature": 0.5,
                "auto_generate_prompt": False
            },
            {
                "agent_name": "Economic Forecaster",
                "description": "Predicts economic trends",
                "system_prompt": "You are an expert in economic forecasting.",
                "model_name": "gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
                "temperature": 0.5,
                "auto_generate_prompt": False
            }
        ],
        "max_loops": 1,
        "swarm_type": "ConcurrentWorkflow",
        "task": "What are the best etfs and index funds for ai and tech?",
        "output_type": "dict"
    }
    
    response = requests.post(
        f"{API_BASE_URL}/v1/swarm/completions",
        headers=headers,
        json=swarm_config
    )
    
    if response.status_code == 200:
        result = response.json()
        print("Swarm completed successfully!")
        print(f"Cost: ${result['metadata']['billing_info']['total_cost']}")
        print(f"Execution time: {result['metadata']['execution_time_seconds']} seconds")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    ```

=== "TypeScript (fetch)"
    ```typescript
    interface AgentSpec {
        agent_name: string;
        description: string;
        system_prompt: string;
        model_name: string;
        role: string;
        max_loops: number;
        max_tokens: number;
        temperature: number;
        auto_generate_prompt: boolean;
    }

    interface SwarmConfig {
        name: string;
        description: string;
        agents: AgentSpec[];
        max_loops: number;
        swarm_type: string;
        task: string;
        output_type: string;
    }

    const API_BASE_URL = "https://api.swarms.world";
    const API_KEY = "your_api_key_here";

    async function runSwarm(): Promise<void> {
        const swarmConfig: SwarmConfig = {
            name: "Financial Analysis Swarm",
            description: "Market analysis swarm",
            agents: [
                {
                    agent_name: "Market Analyst",
                    description: "Analyzes market trends",
                    system_prompt: "You are a financial analyst expert.",
                    model_name: "openai/gpt-4o",
                    role: "worker",
                    max_loops: 1,
                    max_tokens: 8192,
                    temperature: 0.5,
                    auto_generate_prompt: false
                },
                {
                    agent_name: "Economic Forecaster",
                    description: "Predicts economic trends",
                    system_prompt: "You are an expert in economic forecasting.",
                    model_name: "gpt-4o",
                    role: "worker",
                    max_loops: 1,
                    max_tokens: 8192,
                    temperature: 0.5,
                    auto_generate_prompt: false
                }
            ],
            max_loops: 1,
            swarm_type: "ConcurrentWorkflow",
            task: "What are the best etfs and index funds for ai and tech?",
            output_type: "dict"
        };

        try {
            const response = await fetch(`${API_BASE_URL}/v1/swarm/completions`, {
                method: 'POST',
                headers: {
                    'x-api-key': API_KEY,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(swarmConfig)
            });

            if (response.ok) {
                const result = await response.json();
                console.log("Swarm completed successfully!");
                console.log(`Cost: $${result.metadata.billing_info.total_cost}`);
                console.log(`Execution time: ${result.metadata.execution_time_seconds} seconds`);
            } else {
                console.error(`Error: ${response.status} - ${await response.text()}`);
            }
        } catch (error) {
            console.error("Request failed:", error);
        }
    }

    runSwarm();
    ```

**Example Response**:
```json
{
  "status": "success",
  "swarm_name": "financial-analysis-swarm",
  "description": "Analyzes financial data for risk assessment",
  "swarm_type": "SequentialWorkflow",
  "task": "Analyze the provided quarterly financials for Company XYZ and identify potential risk factors. Summarize key insights and provide recommendations for risk mitigation.",
  "output": {
    "financial_analysis": {
      "risk_factors": [...],
      "key_insights": [...],
      "recommendations": [...]
    }
  },
  "metadata": {
    "max_loops": 2,
    "num_agents": 3,
    "execution_time_seconds": 12.45,
    "completion_time": 1709563245.789,
    "billing_info": {
      "cost_breakdown": {
        "agent_cost": 0.03,
        "input_token_cost": 0.002134,
        "output_token_cost": 0.006789,
        "token_counts": {
          "total_input_tokens": 1578,
          "total_output_tokens": 3456,
          "total_tokens": 5034,
          "per_agent": {...}
        },
        "num_agents": 3,
        "execution_time_seconds": 12.45
      },
      "total_cost": 0.038923
    }
  }
}
```

#### Run Batch Completions

Run multiple swarms as a batch operation.

**Endpoint**: `/v1/swarm/batch/completions`  
**Method**: POST  
**Rate Limit**: 100 requests per 60 seconds

**Request Parameters**:

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| swarms | Array<SwarmSpec> | List of swarm specifications | Yes |

=== "Shell (curl)"
    ```bash
    curl -X POST "https://api.swarms.world/v1/swarm/batch/completions" \
      -H "x-api-key: $SWARMS_API_KEY" \
      -H "Content-Type: application/json" \
      -d '[
        {
          "name": "Batch Swarm 1",
          "description": "First swarm in the batch",
          "agents": [
            {
              "agent_name": "Research Agent",
              "description": "Conducts research",
              "system_prompt": "You are a research assistant.",
              "model_name": "gpt-4o",
              "role": "worker",
              "max_loops": 1
            },
            {
              "agent_name": "Analysis Agent",
              "description": "Analyzes data",
              "system_prompt": "You are a data analyst.",
              "model_name": "gpt-4o",
              "role": "worker",
              "max_loops": 1
            }
          ],
          "max_loops": 1,
          "swarm_type": "SequentialWorkflow",
          "task": "Research AI advancements."
        },
        {
          "name": "Batch Swarm 2",
          "description": "Second swarm in the batch",
          "agents": [
            {
              "agent_name": "Writing Agent",
              "description": "Writes content",
              "system_prompt": "You are a content writer.",
              "model_name": "gpt-4o",
              "role": "worker",
              "max_loops": 1
            },
            {
              "agent_name": "Editing Agent",
              "description": "Edits content",
              "system_prompt": "You are an editor.",
              "model_name": "gpt-4o",
              "role": "worker",
              "max_loops": 1
            }
          ],
          "max_loops": 1,
          "swarm_type": "SequentialWorkflow",
          "task": "Write a summary of AI research."
        }
      ]'
    ```

=== "Python (requests)"
    ```python
    import requests
    import json

    API_BASE_URL = "https://api.swarms.world"
    API_KEY = "your_api_key_here"
    
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    
    batch_swarms = [
        {
            "name": "Batch Swarm 1",
            "description": "First swarm in the batch",
            "agents": [
                {
                    "agent_name": "Research Agent",
                    "description": "Conducts research",
                    "system_prompt": "You are a research assistant.",
                    "model_name": "gpt-4o",
                    "role": "worker",
                    "max_loops": 1
                },
                {
                    "agent_name": "Analysis Agent",
                    "description": "Analyzes data",
                    "system_prompt": "You are a data analyst.",
                    "model_name": "gpt-4o",
                    "role": "worker",
                    "max_loops": 1
                }
            ],
            "max_loops": 1,
            "swarm_type": "SequentialWorkflow",
            "task": "Research AI advancements."
        },
        {
            "name": "Batch Swarm 2",
            "description": "Second swarm in the batch",
            "agents": [
                {
                    "agent_name": "Writing Agent",
                    "description": "Writes content",
                    "system_prompt": "You are a content writer.",
                    "model_name": "gpt-4o",
                    "role": "worker",
                    "max_loops": 1
                },
                {
                    "agent_name": "Editing Agent",
                    "description": "Edits content",
                    "system_prompt": "You are an editor.",
                    "model_name": "gpt-4o",
                    "role": "worker",
                    "max_loops": 1
                }
            ],
            "max_loops": 1,
            "swarm_type": "SequentialWorkflow",
            "task": "Write a summary of AI research."
        }
    ]
    
    response = requests.post(
        f"{API_BASE_URL}/v1/swarm/batch/completions",
        headers=headers,
        json=batch_swarms
    )
    
    if response.status_code == 200:
        results = response.json()
        print(f"Batch completed with {len(results)} swarms")
        for i, result in enumerate(results):
            print(f"Swarm {i+1}: {result['swarm_name']} - {result['status']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    ```

=== "TypeScript (fetch)"
    ```typescript
    interface AgentSpec {
        agent_name: string;
        description: string;
        system_prompt: string;
        model_name: string;
        role: string;
        max_loops: number;
    }

    interface SwarmSpec {
        name: string;
        description: string;
        agents: AgentSpec[];
        max_loops: number;
        swarm_type: string;
        task: string;
    }

    const API_BASE_URL = "https://api.swarms.world";
    const API_KEY = "your_api_key_here";

    async function runBatchSwarms(): Promise<void> {
        const batchSwarms: SwarmSpec[] = [
            {
                name: "Batch Swarm 1",
                description: "First swarm in the batch",
                agents: [
                    {
                        agent_name: "Research Agent",
                        description: "Conducts research",
                        system_prompt: "You are a research assistant.",
                        model_name: "gpt-4o",
                        role: "worker",
                        max_loops: 1
                    },
                    {
                        agent_name: "Analysis Agent",
                        description: "Analyzes data",
                        system_prompt: "You are a data analyst.",
                        model_name: "gpt-4o",
                        role: "worker",
                        max_loops: 1
                    }
                ],
                max_loops: 1,
                swarm_type: "SequentialWorkflow",
                task: "Research AI advancements."
            },
            {
                name: "Batch Swarm 2",
                description: "Second swarm in the batch",
                agents: [
                    {
                        agent_name: "Writing Agent",
                        description: "Writes content",
                        system_prompt: "You are a content writer.",
                        model_name: "gpt-4o",
                        role: "worker",
                        max_loops: 1
                    },
                    {
                        agent_name: "Editing Agent",
                        description: "Edits content",
                        system_prompt: "You are an editor.",
                        model_name: "gpt-4o",
                        role: "worker",
                        max_loops: 1
                    }
                ],
                max_loops: 1,
                swarm_type: "SequentialWorkflow",
                task: "Write a summary of AI research."
            }
        ];

        try {
            const response = await fetch(`${API_BASE_URL}/v1/swarm/batch/completions`, {
                method: 'POST',
                headers: {
                    'x-api-key': API_KEY,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(batchSwarms)
            });

            if (response.ok) {
                const results = await response.json();
                console.log(`Batch completed with ${results.length} swarms`);
                results.forEach((result: any, index: number) => {
                    console.log(`Swarm ${index + 1}: ${result.swarm_name} - ${result.status}`);
                });
            } else {
                console.error(`Error: ${response.status} - ${await response.text()}`);
            }
        } catch (error) {
            console.error("Request failed:", error);
        }
    }

    runBatchSwarms();
    ```

**Example Response**:
```json
[
  {
    "status": "success",
    "swarm_name": "risk-analysis",
    "task": "Analyze risk factors for investment portfolio",
    "output": {...},
    "metadata": {...}
  },
  {
    "status": "success",
    "swarm_name": "market-sentiment",
    "task": "Assess current market sentiment for technology sector",
    "output": {...},
    "metadata": {...}
  }
]
```

## Individual Agent Endpoints

### Run Single Agent

Run a single agent with the specified configuration.

**Endpoint**: `/v1/agent/completions`  
**Method**: POST  
**Rate Limit**: 100 requests per 60 seconds

**Request Parameters**:

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| agent_config | AgentSpec | Configuration for the agent | Yes |
| task | string | The task to be completed by the agent | Yes |

=== "Shell (curl)"
    ```bash
    curl -X POST "https://api.swarms.world/v1/agent/completions" \
         -H "x-api-key: your_api_key_here" \
         -H "Content-Type: application/json" \
         -d '{
           "agent_config": {
             "agent_name": "Research Assistant",
             "description": "Helps with research tasks",
             "system_prompt": "You are a research assistant expert.",
             "model_name": "gpt-4o",
             "max_loops": 1,
             "max_tokens": 8192,
             "temperature": 0.5
           },
           "task": "Research the latest developments in quantum computing."
         }'
    ```

=== "Python (requests)"
    ```python
    import requests
    import json

    API_BASE_URL = "https://api.swarms.world"
    API_KEY = "your_api_key_here"
    
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    
    agent_request = {
        "agent_config": {
            "agent_name": "Research Assistant",
            "description": "Helps with research tasks",
            "system_prompt": "You are a research assistant expert.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "max_tokens": 8192,
            "temperature": 0.5
        },
        "task": "Research the latest developments in quantum computing."
    }
    
    response = requests.post(
        f"{API_BASE_URL}/v1/agent/completions",
        headers=headers,
        json=agent_request
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Agent {result['name']} completed successfully!")
        print(f"Usage: {result['usage']['total_tokens']} tokens")
        print(f"Output: {result['outputs']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    ```

=== "TypeScript (fetch)"
    ```typescript
    interface AgentConfig {
        agent_name: string;
        description: string;
        system_prompt: string;
        model_name: string;
        max_loops: number;
        max_tokens: number;
        temperature: number;
    }

    interface AgentRequest {
        agent_config: AgentConfig;
        task: string;
    }

    const API_BASE_URL = "https://api.swarms.world";
    const API_KEY = "your_api_key_here";

    async function runSingleAgent(): Promise<void> {
        const agentRequest: AgentRequest = {
            agent_config: {
                agent_name: "Research Assistant",
                description: "Helps with research tasks",
                system_prompt: "You are a research assistant expert.",
                model_name: "gpt-4o",
                max_loops: 1,
                max_tokens: 8192,
                temperature: 0.5
            },
            task: "Research the latest developments in quantum computing."
        };

        try {
            const response = await fetch(`${API_BASE_URL}/v1/agent/completions`, {
                method: 'POST',
                headers: {
                    'x-api-key': API_KEY,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(agentRequest)
            });

            if (response.ok) {
                const result = await response.json();
                console.log(`Agent ${result.name} completed successfully!`);
                console.log(`Usage: ${result.usage.total_tokens} tokens`);
                console.log(`Output:`, result.outputs);
            } else {
                console.error(`Error: ${response.status} - ${await response.text()}`);
            }
        } catch (error) {
            console.error("Request failed:", error);
        }
    }

    runSingleAgent();
    ```

**Example Response**:
```json
{
  "id": "agent-abc123",
  "success": true,
  "name": "Research Assistant",
  "description": "Helps with research tasks",
  "temperature": 0.5,
  "outputs": {},
  "usage": {
    "input_tokens": 150,
    "output_tokens": 450,
    "total_tokens": 600
  },
  "timestamp": "2024-03-05T12:34:56.789Z"
}
```

### AgentCompletion Model

The `AgentCompletion` model defines the configuration for running a single agent task.

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| `agent_config` | AgentSpec | The configuration of the agent to be completed | Yes |
| `task` | string | The task to be completed by the agent | Yes |
| `history` | Dict[str, Any] | The history of the agent's previous tasks and responses | No |


### AgentSpec Model

The `AgentSpec` model defines the configuration for an individual agent.

| Field | Type | Default | Description | Required |
|-------|------|---------|-------------|----------|
| `agent_name` | string | None | The unique name assigned to the agent | Yes |
| `description` | string | None | Detailed explanation of the agent's purpose | No |
| `system_prompt` | string | None | Initial instruction provided to the agent | No |
| `model_name` | string | "gpt-4o-mini" | Name of the AI model to use | No |
| `auto_generate_prompt` | boolean | false | Whether to auto-generate prompts | No |
| `max_tokens` | integer | 8192 | Maximum tokens in response | No |
| `temperature` | float | 0.5 | Controls randomness (0-1) | No |
| `role` | string | "worker" | Role of the agent | No |
| `max_loops` | integer | 1 | Maximum iterations | No |
| `tools_list_dictionary` | List[Dict] | None | Available tools for the agent | No |
| `mcp_url` | string | None | URL for Model Control Protocol | No |


Execute a task using a single agent with specified configuration.

**Endpoint**: `/v1/agent/completions`  
**Method**: POST  
**Rate Limit**: 100 requests per 60 seconds

**Request Body**:
```json
{
  "agent_config": {
    "agent_name": "Research Assistant",
    "description": "Specialized in research and analysis",
    "system_prompt": "You are an expert research assistant.",
    "model_name": "gpt-4o",
    "auto_generate_prompt": false,
    "max_tokens": 8192,
    "temperature": 0.5,
    "role": "worker",
    "max_loops": 1,
    "tools_list_dictionary": [
      {
        "name": "search",
        "description": "Search the web for information",
        "parameters": {
          "query": "string"
        }
      }
    ],
    "mcp_url": "https://example-mcp.com"
  },
  "task": "Research the latest developments in quantum computing and summarize key findings",
  "history": {
    "previous_research": "Earlier findings on quantum computing basics...",
    "user_preferences": "Focus on practical applications..."
  }
}
```

**Response**:
```json
{
  "id": "agent-abc123xyz",
  "success": true,
  "name": "Research Assistant",
  "description": "Specialized in research and analysis",
  "temperature": 0.5,
  "outputs": {
    "research_summary": "...",
    "key_findings": [
      "..."
    ]
  },
  "usage": {
    "input_tokens": 450,
    "output_tokens": 850,
    "total_tokens": 1300,
    "mcp_url": 0.1
  },
  "timestamp": "2024-03-05T12:34:56.789Z"
}
```

#### Run Batch Agents

Execute multiple agent tasks in parallel.

**Endpoint**: `/v1/agent/batch/completions`  
**Method**: POST  
**Rate Limit**: 100 requests per 60 seconds  
**Maximum Batch Size**: 10 requests
**Input** A list of `AgentCompeletion` inputs

=== "Shell (curl)"
    ```bash
    curl -X POST "https://api.swarms.world/v1/agent/batch/completions" \
         -H "x-api-key: your_api_key_here" \
         -H "Content-Type: application/json" \
         -d '[
           {
             "agent_config": {
               "agent_name": "Market Analyst",
               "description": "Expert in market analysis",
               "system_prompt": "You are a financial market analyst.",
               "model_name": "gpt-4o",
               "temperature": 0.3
             },
             "task": "Analyze the current market trends in AI technology sector"
           },
           {
             "agent_config": {
               "agent_name": "Technical Writer",
               "description": "Specialized in technical documentation",
               "system_prompt": "You are a technical documentation expert.",
               "model_name": "gpt-4o",
               "temperature": 0.7
             },
             "task": "Create a technical guide for implementing OAuth2 authentication"
           }
         ]'
    ```

=== "Python (requests)"
    ```python
    import requests
    import json

    API_BASE_URL = "https://api.swarms.world"
    API_KEY = "your_api_key_here"
    
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    
    batch_agents = [
        {
            "agent_config": {
                "agent_name": "Market Analyst",
                "description": "Expert in market analysis",
                "system_prompt": "You are a financial market analyst.",
                "model_name": "gpt-4o",
                "temperature": 0.3
            },
            "task": "Analyze the current market trends in AI technology sector"
        },
        {
            "agent_config": {
                "agent_name": "Technical Writer",
                "description": "Specialized in technical documentation",
                "system_prompt": "You are a technical documentation expert.",
                "model_name": "gpt-4o",
                "temperature": 0.7
            },
            "task": "Create a technical guide for implementing OAuth2 authentication"
        }
    ]
    
    response = requests.post(
        f"{API_BASE_URL}/v1/agent/batch/completions",
        headers=headers,
        json=batch_agents
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Batch completed with {result['total_requests']} agents")
        print(f"Execution time: {result['execution_time']} seconds")
        print("\nResults:")
        for i, agent_result in enumerate(result['results']):
            print(f"  Agent {i+1}: {agent_result['name']} - {agent_result['success']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    ```

=== "TypeScript (fetch)"
    ```typescript
    interface AgentConfig {
        agent_name: string;
        description: string;
        system_prompt: string;
        model_name: string;
        temperature: number;
    }

    interface AgentCompletion {
        agent_config: AgentConfig;
        task: string;
    }

    const API_BASE_URL = "https://api.swarms.world";
    const API_KEY = "your_api_key_here";

    async function runBatchAgents(): Promise<void> {
        const batchAgents: AgentCompletion[] = [
            {
                agent_config: {
                    agent_name: "Market Analyst",
                    description: "Expert in market analysis",
                    system_prompt: "You are a financial market analyst.",
                    model_name: "gpt-4o",
                    temperature: 0.3
                },
                task: "Analyze the current market trends in AI technology sector"
            },
            {
                agent_config: {
                    agent_name: "Technical Writer",
                    description: "Specialized in technical documentation",
                    system_prompt: "You are a technical documentation expert.",
                    model_name: "gpt-4o",
                    temperature: 0.7
                },
                task: "Create a technical guide for implementing OAuth2 authentication"
            }
        ];

        try {
            const response = await fetch(`${API_BASE_URL}/v1/agent/batch/completions`, {
                method: 'POST',
                headers: {
                    'x-api-key': API_KEY,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(batchAgents)
            });

            if (response.ok) {
                const result = await response.json();
                console.log(`Batch completed with ${result.total_requests} agents`);
                console.log(`Execution time: ${result.execution_time} seconds`);
                console.log("\nResults:");
                result.results.forEach((agentResult: any, index: number) => {
                    console.log(`  Agent ${index + 1}: ${agentResult.name} - ${agentResult.success}`);
                });
            } else {
                console.error(`Error: ${response.status} - ${await response.text()}`);
            }
        } catch (error) {
            console.error("Request failed:", error);
        }
    }

    runBatchAgents();
    ```

**Response**:
```json
{
  "batch_id": "agent-batch-xyz789",
  "total_requests": 2,
  "execution_time": 15.5,
  "timestamp": "2024-03-05T12:34:56.789Z",
  "results": [
    {
      "id": "agent-abc123",
      "success": true,
      "name": "Market Analyst",
      "outputs": {
        "market_analysis": "..."
      },
      "usage": {
        "input_tokens": 300,
        "output_tokens": 600,
        "total_tokens": 900
      }
    },
    {
      "id": "agent-def456",
      "success": true,
      "name": "Technical Writer",
      "outputs": {
        "technical_guide": "..."
      },
      "usage": {
        "input_tokens": 400,
        "output_tokens": 800,
        "total_tokens": 1200
      }
    }
  ]
}
```

-----

## Production Examples

## Error Handling

The Swarms API follows standard HTTP status codes for error responses:

| Status Code | Meaning | Handling Strategy |
|-------------|---------|-------------------|
| 400 | Bad Request | Validate request parameters before sending |
| 401 | Unauthorized | Check API key validity |
| 403 | Forbidden | Verify API key permissions |
| 404 | Not Found | Check endpoint URL and resource IDs |
| 429 | Too Many Requests | Implement exponential backoff retry logic |
| 500 | Internal Server Error | Retry with backoff, then contact support |

Error responses include a detailed message explaining the issue:

```json
{
  "detail": "Failed to create swarm: Invalid swarm_type specified"
}
```

## Rate Limiting

| Description | Details |
|-------------|---------|
| Rate Limit | 100 requests per 60-second window |
| Exceed Consequence | 429 status code returned |
| Recommended Action | Implement retry logic with exponential backoff |

## Billing & Cost Management

| Cost Factor | Description |
|-------------|-------------|
| Agent Count | Base cost per agent |
| Input Tokens | Cost based on size of input data and prompts |
| Output Tokens | Cost based on length of generated responses |
| Time of Day | Reduced rates during nighttime hours (8 PM to 6 AM PT) |
| Cost Information | Included in each response's metadata |

## Best Practices

### Task Description

| Practice | Description |
|----------|-------------|
| Detail | Provide detailed, specific task descriptions |
| Context | Include all necessary context and constraints |
| Structure | Structure complex inputs for easier processing |

### Agent Configuration

| Practice | Description |
|----------|-------------|
| Simple Tasks | Use `AutoSwarmBuilder` for automatic agent generation |
| Complex Tasks | Manually define agents with specific expertise |
| Workflow | Use appropriate `swarm_type` for your workflow pattern |

### Production Implementation

| Practice | Description |
|----------|-------------|
| Error Handling | Implement robust error handling and retries |
| Logging | Log API responses for debugging and auditing |
| Cost Monitoring | Monitor costs closely during development and testing |

### Cost Optimization

| Practice | Description |
|----------|-------------|
| Batching | Batch related tasks when possible |
| Scheduling | Schedule non-urgent tasks during discount hours |
| Scoping | Carefully scope task descriptions to reduce token usage |
| Caching | Cache results when appropriate |

## Support

| Support Type | Contact Information |
|--------------|---------------------|
| Documentation | [https://docs.swarms.world](https://docs.swarms.world) |
| Email | kye@swarms.world |
| Community | [https://discord.gg/EamjgSaEQf](https://discord.gg/EamjgSaEQf) |
| Marketplace | [https://swarms.world](https://swarms.world) |
| Website | [https://swarms.ai](https://swarms.ai) |

## Service Tiers

### Standard Tier

| Feature | Description |
|---------|-------------|
| Processing | Default processing tier |
| Execution | Immediate execution |
| Priority | Higher priority processing |
| Pricing | Standard pricing |
| Timeout | 5-minute timeout limit |

### Flex Tier

| Feature | Description |
|---------|-------------|
| Cost | Lower cost processing |
| Retries | Automatic retries (up to 3 attempts) |
| Timeout | 15-minute timeout |
| Discount | 75% discount on token costs |
| Suitability | Best for non-urgent tasks |
| Backoff | Exponential backoff on resource contention |
| Configuration | Set `service_tier: "flex"` in SwarmSpec |