# Swarms API Documentation

The Swarms API is a powerful REST API designed to help you create, manage, and execute various types of swarms efficiently. Whether you need to run tasks sequentially, concurrently, or in a custom workflow, the Swarms API has you covered.

### Key Features:
- **Sequential Swarms**: Execute tasks one after another in a defined order.
- **Concurrent Swarms**: Run multiple tasks simultaneously to save time and resources.
- **Custom Workflows**: Design your own swarm workflows to fit your specific needs.

To get started, find your API key in the Swarms Cloud dashboard. [Get your API key here](https://swarms.world/platform/api-keys)

## Base URL
```
https://swarms-api-285321057562.us-east1.run.app
```

## Authentication
All API requests (except `/health`) require authentication using an API key passed in the `x-api-key` header:

```http
x-api-key: your_api_key_here
```

## Endpoints

### Health Check
Check if the API is operational.

**Endpoint:** `GET /health`  
**Authentication Required:** No  
**Response:**
```json
{
    "status": "ok"
}
```

### Single Swarm Completion
Run a single swarm with specified agents and tasks.

**Endpoint:** `POST /v1/swarm/completions`  
**Authentication Required:** Yes

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| name | string | Optional | "swarms-01" | Name of the swarm (max 100 chars) |
| description | string | Optional | - | Description of the swarm (max 500 chars) |
| agents | array | Required | - | Array of agent configurations |
| max_loops | integer | Optional | 1 | Maximum number of iterations |
| swarm_type | string | Optional | - | Type of swarm workflow |
| task | string | Required | - | The task to be performed |
| img | string | Optional | - | Image URL if relevant |
| return_history | boolean | Optional | true | Whether to return the full conversation history |
| rules | string | Optional | - | Rules for the swarm to follow |
| rearrange_flow | string | Optional | - | Flow pattern for agent rearrangement |
| output_type | string | Optional | "str" | Output format ("str", "json", "dict", "yaml", "list") |
| schedule | object | Optional | - | Scheduling information for the swarm |

#### Schedule Configuration Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| scheduled_time | datetime | Required | - | When to execute the swarm (UTC) |
| timezone | string | Optional | "UTC" | Timezone for the scheduled time |

#### Agent Configuration Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| agent_name | string | Required | - | Name of the agent (max 100 chars) |
| description | string | Optional | - | Description of the agent (max 500 chars) |
| system_prompt | string | Optional | - | System prompt for the agent (max 500 chars) |
| model_name | string | Optional | "gpt-4o" | Model to be used by the agent |
| auto_generate_prompt | boolean | Optional | false | Whether to auto-generate prompts |
| max_tokens | integer | Optional | - | Maximum tokens for response |
| temperature | float | Optional | 0.5 | Temperature for response generation |
| role | string | Optional | "worker" | Role of the agent |
| max_loops | integer | Optional | 1 | Maximum iterations for this agent |

## Available Swarm Types

| Swarm Type | Description |
|------------|-------------|
| AgentRearrange | Rearranges agents dynamically to optimize task execution |
| MixtureOfAgents | Combines different agents to leverage their unique capabilities |
| SpreadSheetSwarm | Utilizes spreadsheet-like operations for data manipulation |
| SequentialWorkflow | Executes tasks in a predefined sequential order |
| ConcurrentWorkflow | Runs tasks concurrently to improve efficiency |
| GroupChat | Facilitates communication among agents in a chat format |
| MultiAgentRouter | Routes tasks to agents based on their expertise |
| AutoSwarmBuilder | Automatically constructs swarms based on task requirements |
| HiearchicalSwarm | Organizes agents in a hierarchy for complex tasks |
| auto | Automatically selects the most suitable swarm type |
| MajorityVoting | Uses majority voting to reach consensus on outcomes |

## Job Scheduling Endpoints

### Schedule a Swarm
Schedule a swarm to run at a specific time.

**Endpoint:** `POST /v1/swarm/schedule`  
**Authentication Required:** Yes

#### Request Format
Same as single swarm completion, with additional `schedule` object:

```json
{
    "name": "Scheduled Swarm",
    "agents": [...],
    "task": "Perform analysis",
    "schedule": {
        "scheduled_time": "2024-03-20T15:00:00Z",
        "timezone": "America/New_York"
    }
}
```

### List Scheduled Jobs
Get all scheduled swarm jobs.

**Endpoint:** `GET /v1/swarm/schedule`  
**Authentication Required:** Yes

#### Response Format
```json
{
    "status": "success",
    "scheduled_jobs": [
        {
            "job_id": "swarm_analysis_1234567890",
            "swarm_name": "Analysis Swarm",
            "scheduled_time": "2024-03-20T15:00:00Z",
            "timezone": "America/New_York"
        }
    ]
}
```

### Cancel Scheduled Job
Cancel a scheduled swarm job.

**Endpoint:** `DELETE /v1/swarm/schedule/{job_id}`  
**Authentication Required:** Yes

#### Response Format
```json
{
    "status": "success",
    "message": "Scheduled job cancelled successfully",
    "job_id": "swarm_analysis_1234567890"
}
```

## Billing and Credits

The API uses a credit-based billing system with the following components:

### Cost Calculation

| Component | Cost |
|-----------|------|
| Base cost per agent | $0.01 |
| Input tokens (per 1M) | $2.00 |
| Output tokens (per 1M) | $6.00 |

Special pricing:
- California night time hours (8 PM to 6 AM PT): 75% discount on token costs
- Credits are deducted in the following order:
  1. Free credits
  2. Regular credits

Costs are calculated based on:
- Number of agents used
- Total input tokens (including system prompts and agent memory)
- Total output tokens generated
- Execution time

## Error Handling

| HTTP Status Code | Description |
|-----------------|-------------|
| 402 | Insufficient credits |
| 403 | Invalid API key |
| 404 | Resource not found |
| 500 | Internal server error |

## Best Practices

1. Start with small swarms and gradually increase complexity
2. Monitor credit usage and token counts
3. Use appropriate max_loops values to control execution
4. Implement proper error handling for API responses
5. Consider using batch completions for multiple related tasks

## Response Structures

### Single Swarm Response

```json
{
    "status": "success",
    "swarm_name": "Test Swarm",
    "description": "A test swarm",
    "swarm_type": "ConcurrentWorkflow",
    "task": "Write a blog post",
    "output": {
        // Swarm output here
    },
    "metadata": {
        "max_loops": 1,
        "num_agents": 2,
        "execution_time_seconds": 5.23,
        "completion_time": 1647123456.789,
        "billing_info": {
            "cost_breakdown": {
                "agent_cost": 0.02,
                "input_token_cost": 0.015,
                "output_token_cost": 0.045,
                "token_counts": {
                    "total_input_tokens": 1500,
                    "total_output_tokens": 3000,
                    "total_tokens": 4500,
                    "per_agent": {
                        "agent1": {
                            "input_tokens": 750,
                            "output_tokens": 1500,
                            "total_tokens": 2250
                        },
                        "agent2": {
                            "input_tokens": 750,
                            "output_tokens": 1500,
                            "total_tokens": 2250
                        }
                    }
                },
                "num_agents": 2,
                "execution_time_seconds": 5.23
            },
            "total_cost": 0.08
        }
    }
}
```

### Batch Swarm Response

```json
[
    {
        "status": "success",
        "swarm_name": "Batch Swarm 1",
        "output": {},
        "metadata": {}
    },
    {
        "status": "success",
        "swarm_name": "Batch Swarm 2",
        "output": {},
        "metadata": {}
    }
]
```

## Logs Endpoint

### Get Swarm Logs
Retrieve execution logs for your API key.

**Endpoint:** `GET /v1/swarm/logs`  
**Authentication Required:** Yes

#### Response Format
```json
{
    "status": "success",
    "count": 2,
    "logs": [
        {
            "api_key": "masked",
            "data": {
                "swarm_name": "Test Swarm",
                "task": "Write a blog post",
                "execution_time": "2024-03-19T15:30:00Z",
                "status": "success"
            }
        }
    ]
}
```

## Error Handling

The API uses standard HTTP status codes and provides detailed error messages:

| HTTP Status Code | Description | Example Response |
|-----------------|-------------|------------------|
| 400 | Bad Request - Invalid parameters | `{"detail": "Invalid swarm configuration"}` |
| 401 | Unauthorized - Missing API key | `{"detail": "API key is required"}` |
| 402 | Payment Required - Insufficient credits | `{"detail": "Insufficient credits"}` |
| 403 | Forbidden - Invalid API key | `{"detail": "Invalid API key"}` |
| 429 | Too Many Requests - Rate limit exceeded | `{"detail": "Rate limit exceeded"}` |
| 500 | Internal Server Error | `{"detail": "Internal server error"}` |

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Rate Limit:** 100 requests per minute per IP address
- **Time Window:** 60 seconds
- **Response on Limit Exceeded:** HTTP 429 with retry-after header

# Code Examples

## Python
### Using requests

```python
import requests
from datetime import datetime, timedelta
import pytz

API_KEY = "your_api_key_here"
BASE_URL = "https://swarms-api-285321057562.us-east1.run.app"

headers = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}

def run_single_swarm():
    payload = {
        "name": "Financial Analysis Swarm",
        "description": "Market analysis swarm",
        "agents": [
            {
                "agent_name": "Market Analyst",
                "description": "Analyzes market trends",
                "system_prompt": "You are a financial analyst expert.",
                "model_name": "gpt-4o",
                "role": "worker",
                "max_loops": 1
            }
        ],
        "max_loops": 1,
        "swarm_type": "SequentialWorkflow",
        "task": "Analyze current market trends in tech sector",
        "return_history": True,
        "rules": "Focus on major market indicators"
    }
    
    response = requests.post(
        f"{BASE_URL}/v1/swarm/completions",
        headers=headers,
        json=payload
    )
    
    return response.json()

def schedule_swarm():
    # Schedule for 1 hour from now
    scheduled_time = datetime.now(pytz.UTC) + timedelta(hours=1)
    
    payload = {
        "name": "Scheduled Analysis",
        "agents": [
            {
                "agent_name": "Analyst",
                "system_prompt": "You are a market analyst.",
                "model_name": "gpt-4o",
                "role": "worker"
            }
        ],
        "task": "Analyze tech trends",
        "schedule": {
            "scheduled_time": scheduled_time.isoformat(),
            "timezone": "America/New_York"
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/v1/swarm/schedule",
        headers=headers,
        json=payload
    )
    
    return response.json()

def get_scheduled_jobs():
    response = requests.get(
        f"{BASE_URL}/v1/swarm/schedule",
        headers=headers
    )
    return response.json()

def cancel_scheduled_job(job_id: str):
    response = requests.delete(
        f"{BASE_URL}/v1/swarm/schedule/{job_id}",
        headers=headers
    )
    return response.json()

def get_swarm_logs():
    response = requests.get(
        f"{BASE_URL}/v1/swarm/logs",
        headers=headers
    )
    return response.json()
```

## Node.js
### Using Fetch API

```javascript
const API_KEY = 'your_api_key_here';
const BASE_URL = 'https://swarms-api-285321057562.us-east1.run.app';

const headers = {
  'x-api-key': API_KEY,
  'Content-Type': 'application/json'
};

// Schedule a swarm
async function scheduleSwarm() {
  const scheduledTime = new Date();
  scheduledTime.setHours(scheduledTime.getHours() + 1);

  const payload = {
    name: 'Scheduled Analysis',
    agents: [{
      agent_name: 'Analyst',
      system_prompt: 'You are a market analyst.',
      model_name: 'gpt-4o',
      role: 'worker'
    }],
    task: 'Analyze tech trends',
    schedule: {
      scheduled_time: scheduledTime.toISOString(),
      timezone: 'America/New_York'
    }
  };

  try {
    const response = await fetch(`${BASE_URL}/v1/swarm/schedule`, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload)
    });
    
    return await response.json();
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
}

// Get scheduled jobs
async function getScheduledJobs() {
  try {
    const response = await fetch(`${BASE_URL}/v1/swarm/schedule`, {
      headers
    });
    return await response.json();
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
}

// Cancel scheduled job
async function cancelScheduledJob(jobId) {
  try {
    const response = await fetch(`${BASE_URL}/v1/swarm/schedule/${jobId}`, {
      method: 'DELETE',
      headers
    });
    return await response.json();
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
}
```

## Shell (cURL)

### Schedule a Swarm

```bash
curl -X POST "https://swarms-api-285321057562.us-east1.run.app/v1/swarm/schedule" \
  -H "x-api-key: your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Scheduled Analysis",
    "agents": [
      {
        "agent_name": "Analyst",
        "system_prompt": "You are a market analyst.",
        "model_name": "gpt-4o",
        "role": "worker"
      }
    ],
    "task": "Analyze tech trends",
    "schedule": {
      "scheduled_time": "2024-03-20T15:00:00Z",
      "timezone": "America/New_York"
    }
  }'
```

### Get Scheduled Jobs

```bash
curl -X GET "https://swarms-api-285321057562.us-east1.run.app/v1/swarm/schedule" \
  -H "x-api-key: your_api_key_here"
```

### Cancel Scheduled Job

```bash
curl -X DELETE "https://swarms-api-285321057562.us-east1.run.app/v1/swarm/schedule/job_id_here" \
  -H "x-api-key: your_api_key_here"
```

### Get Swarm Logs

```bash
curl -X GET "https://swarms-api-285321057562.us-east1.run.app/v1/swarm/logs" \
  -H "x-api-key: your_api_key_here"
```