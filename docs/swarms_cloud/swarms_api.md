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

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| name | string | Optional | Name of the swarm (max 100 chars) |
| description | string | Optional | Description of the swarm (max 500 chars) |
| agents | array | Required | Array of agent configurations |
| max_loops | integer | Optional | Maximum number of iterations |
| swarm_type | string | Optional | Type of swarm workflow |
| task | string | Required | The task to be performed |
| img | string | Optional | Image URL if relevant |

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

#### Example Request
```json
{
    "name": "Test Swarm",
    "description": "A test swarm",
    "agents": [
        {
            "agent_name": "Research Agent",
            "description": "Conducts research",
            "system_prompt": "You are a research assistant.",
            "model_name": "gpt-4o",
            "role": "worker",
            "max_loops": 1
        }
    ],
    "max_loops": 1,
    "swarm_type": "ConcurrentWorkflow",
    "task": "Write a short blog post about AI agents."
}
```

#### Response Structure

| Field | Type | Description |
|-------|------|-------------|
| status | string | Status of the swarm execution |
| swarm_name | string | Name of the executed swarm |
| description | string | Description of the swarm |
| task | string | Original task description |
| metadata | object | Execution metadata |
| output | object/array | Results from the swarm execution |

### Batch Swarm Completion
Run multiple swarms in a single request.

**Endpoint:** `POST /v1/swarm/batch/completions`  
**Authentication Required:** Yes

#### Request Format
Array of swarm configurations, each following the same format as single swarm completion.

#### Example Batch Request
```json
[
    {
        "name": "Batch Swarm 1",
        "description": "First swarm in batch",
        "agents": [...],
        "task": "Task 1"
    },
    {
        "name": "Batch Swarm 2",
        "description": "Second swarm in batch",
        "agents": [...],
        "task": "Task 2"
    }
]
```
# Swarms API Implementation Examples

## Python
### Using requests

```python
import requests
import json

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
        "task": "Analyze current market trends in tech sector"
    }
    
    response = requests.post(
        f"{BASE_URL}/v1/swarm/completions",
        headers=headers,
        json=payload
    )
    
    return response.json()

def run_batch_swarms():
    payload = [
        {
            "name": "Market Analysis",
            "description": "First swarm",
            "agents": [
                {
                    "agent_name": "Analyst",
                    "system_prompt": "You are a market analyst.",
                    "model_name": "gpt-4o",
                    "role": "worker"
                }
            ],
            "task": "Analyze tech trends"
        },
        {
            "name": "Risk Assessment",
            "description": "Second swarm",
            "agents": [
                {
                    "agent_name": "Risk Analyst",
                    "system_prompt": "You are a risk analyst.",
                    "model_name": "gpt-4o",
                    "role": "worker"
                }
            ],
            "task": "Assess market risks"
        }
    ]
    
    response = requests.post(
        f"{BASE_URL}/v1/swarm/batch/completions",
        headers=headers,
        json=payload
    )
    
    return response.json()

# Using async/await with aiohttp
import aiohttp
import asyncio

async def run_swarm_async():
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{BASE_URL}/v1/swarm/completions",
            headers=headers,
            json=payload
        ) as response:
            return await response.json()
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

// Single swarm execution
async function runSingleSwarm() {
  const payload = {
    name: 'Financial Analysis',
    description: 'Market analysis swarm',
    agents: [
      {
        agent_name: 'Market Analyst',
        description: 'Analyzes market trends',
        system_prompt: 'You are a financial analyst expert.',
        model_name: 'gpt-4o',
        role: 'worker',
        max_loops: 1
      }
    ],
    max_loops: 1,
    swarm_type: 'SequentialWorkflow',
    task: 'Analyze current market trends'
  };

  try {
    const response = await fetch(`${BASE_URL}/v1/swarm/completions`, {
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

// Batch swarm execution
async function runBatchSwarms() {
  const payload = [
    {
      name: 'Market Analysis',
      agents: [{
        agent_name: 'Analyst',
        system_prompt: 'You are a market analyst.',
        model_name: 'gpt-4o',
        role: 'worker'
      }],
      task: 'Analyze tech trends'
    },
    {
      name: 'Risk Assessment',
      agents: [{
        agent_name: 'Risk Analyst',
        system_prompt: 'You are a risk analyst.',
        model_name: 'gpt-4o',
        role: 'worker'
      }],
      task: 'Assess market risks'
    }
  ];

  try {
    const response = await fetch(`${BASE_URL}/v1/swarm/batch/completions`, {
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
```

### Using Axios

```javascript
const axios = require('axios');

const api = axios.create({
  baseURL: BASE_URL,
  headers: {
    'x-api-key': API_KEY,
    'Content-Type': 'application/json'
  }
});

async function runSwarm() {
  try {
    const response = await api.post('/v1/swarm/completions', payload);
    return response.data;
  } catch (error) {
    console.error('Error:', error.response?.data || error.message);
    throw error;
  }
}
```

## Go

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io/ioutil"
    "net/http"
)

const (
    baseURL = "https://swarms-api-285321057562.us-east1.run.app"
    apiKey  = "your_api_key_here"
)

type Agent struct {
    AgentName    string `json:"agent_name"`
    Description  string `json:"description"`
    SystemPrompt string `json:"system_prompt"`
    ModelName    string `json:"model_name"`
    Role         string `json:"role"`
    MaxLoops     int    `json:"max_loops"`
}

type SwarmRequest struct {
    Name        string   `json:"name"`
    Description string   `json:"description"`
    Agents      []Agent  `json:"agents"`
    MaxLoops    int      `json:"max_loops"`
    SwarmType   string   `json:"swarm_type"`
    Task        string   `json:"task"`
}

func runSingleSwarm() ([]byte, error) {
    payload := SwarmRequest{
        Name:        "Financial Analysis",
        Description: "Market analysis swarm",
        Agents: []Agent{
            {
                AgentName:    "Market Analyst",
                Description:  "Analyzes market trends",
                SystemPrompt: "You are a financial analyst expert.",
                ModelName:    "gpt-4o",
                Role:         "worker",
                MaxLoops:     1,
            },
        },
        MaxLoops:  1,
        SwarmType: "SequentialWorkflow",
        Task:      "Analyze current market trends",
    }

    jsonPayload, err := json.Marshal(payload)
    if err != nil {
        return nil, err
    }

    client := &http.Client{}
    req, err := http.NewRequest("POST", baseURL+"/v1/swarm/completions", bytes.NewBuffer(jsonPayload))
    if err != nil {
        return nil, err
    }

    req.Header.Set("x-api-key", apiKey)
    req.Header.Set("Content-Type", "application/json")

    resp, err := client.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    return ioutil.ReadAll(resp.Body)
}

func main() {
    response, err := runSingleSwarm()
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }
    
    fmt.Printf("Response: %s\n", response)
}
```

## Rust

```rust
use reqwest::{Client, header};
use serde::{Deserialize, Serialize};
use serde_json::json;

const BASE_URL: &str = "https://swarms-api-285321057562.us-east1.run.app";
const API_KEY: &str = "your_api_key_here";

#[derive(Serialize, Deserialize)]
struct Agent {
    agent_name: String,
    description: String,
    system_prompt: String,
    model_name: String,
    role: String,
    max_loops: i32,
}

#[derive(Serialize, Deserialize)]
struct SwarmRequest {
    name: String,
    description: String,
    agents: Vec<Agent>,
    max_loops: i32,
    swarm_type: String,
    task: String,
}

async fn run_single_swarm() -> Result<String, Box<dyn std::error::Error>> {
    let client = Client::new();
    
    let payload = SwarmRequest {
        name: "Financial Analysis".to_string(),
        description: "Market analysis swarm".to_string(),
        agents: vec![Agent {
            agent_name: "Market Analyst".to_string(),
            description: "Analyzes market trends".to_string(),
            system_prompt: "You are a financial analyst expert.".to_string(),
            model_name: "gpt-4o".to_string(),
            role: "worker".to_string(),
            max_loops: 1,
        }],
        max_loops: 1,
        swarm_type: "SequentialWorkflow".to_string(),
        task: "Analyze current market trends".to_string(),
    };

    let response = client
        .post(format!("{}/v1/swarm/completions", BASE_URL))
        .header("x-api-key", API_KEY)
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await?;

    let result = response.text().await?;
    Ok(result)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let response = run_single_swarm().await?;
    println!("Response: {}", response);
    Ok(())
}
```

## C#

```csharp
using System;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

public class SwarmClient
{
    private readonly HttpClient _client;
    private const string BaseUrl = "https://swarms-api-285321057562.us-east1.run.app";
    private readonly string _apiKey;

    public SwarmClient(string apiKey)
    {
        _apiKey = apiKey;
        _client = new HttpClient();
        _client.DefaultRequestHeaders.Add("x-api-key", apiKey);
    }

    public class Agent
    {
        public string AgentName { get; set; }
        public string Description { get; set; }
        public string SystemPrompt { get; set; }
        public string ModelName { get; set; }
        public string Role { get; set; }
        public int MaxLoops { get; set; }
    }

    public class SwarmRequest
    {
        public string Name { get; set; }
        public string Description { get; set; }
        public List<Agent> Agents { get; set; }
        public int MaxLoops { get; set; }
        public string SwarmType { get; set; }
        public string Task { get; set; }
    }

    public async Task<string> RunSingleSwarm()
    {
        var payload = new SwarmRequest
        {
            Name = "Financial Analysis",
            Description = "Market analysis swarm",
            Agents = new List<Agent>
            {
                new Agent
                {
                    AgentName = "Market Analyst",
                    Description = "Analyzes market trends",
                    SystemPrompt = "You are a financial analyst expert.",
                    ModelName = "gpt-4o",
                    Role = "worker",
                    MaxLoops = 1
                }
            },
            MaxLoops = 1,
            SwarmType = "SequentialWorkflow",
            Task = "Analyze current market trends"
        };

        var content = new StringContent(
            JsonSerializer.Serialize(payload),
            Encoding.UTF8,
            "application/json"
        );

        var response = await _client.PostAsync(
            $"{BaseUrl}/v1/swarm/completions",
            content
        );

        return await response.Content.ReadAsStringAsync();
    }
}

// Usage
class Program
{
    static async Task Main(string[] args)
    {
        var client = new SwarmClient("your_api_key_here");
        var response = await client.RunSingleSwarm();
        Console.WriteLine($"Response: {response}");
    }
}
```

## Shell (cURL)

```bash
# Single swarm execution
curl -X POST "https://swarms-api-285321057562.us-east1.run.app/v1/swarm/completions" \
  -H "x-api-key: your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Financial Analysis",
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
    "task": "Analyze current market trends"
  }'

# Batch swarm execution
curl -X POST "https://swarms-api-285321057562.us-east1.run.app/v1/swarm/batch/completions" \
  -H "x-api-key: your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "name": "Market Analysis",
      "agents": [{
        "agent_name": "Analyst",
        "system_prompt": "You are a market analyst.",
        "model_name": "gpt-4o",
        "role": "worker"
      }],
      "task": "Analyze tech trends"
    },
    {
      "name": "Risk Assessment",
      "agents": [{
        "agent_name": "Risk Analyst",
        "system_prompt": "You are a risk analyst.",
        "model_name": "gpt-4o",
        "role": "worker"
      }],
      "task": "Assess market risks"
    }
  ]'
```


## Billing and Credits

The API uses a credit-based billing system with the following components:

### Cost Calculation

| Component | Cost |
|-----------|------|
| Base cost per agent | $0.01 |
| Input tokens (per 1M) | $5.00 |
| Output tokens (per 1M) | $15.50 |

Credits are deducted based on:
- Number of agents used
- Total input tokens (including system prompts and agent memory)
- Total output tokens generated
- Execution time

### Credit Types
- Free credits: Used first
- Regular credits: Used after free credits are exhausted

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