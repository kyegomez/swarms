# Swarms API with Tools Guide


Swarms API allows you to create and manage AI agent swarms with optional tool integration. This guide will walk you through setting up and using the Swarms API with tools.

## Prerequisites

- Python 3.7+
- Swarms API key
- Required Python packages:
  - `requests`
  - `python-dotenv`

## Installation & Setup

1. Install required packages:

```bash
pip install requests python-dotenv
```

2. Create a `.env` file in your project root:

```bash
SWARMS_API_KEY=your_api_key_here
```

3. Basic setup code:

```python
import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

API_KEY = os.getenv("SWARMS_API_KEY")
BASE_URL = "https://api.swarms.world"

headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
```

## Creating a Swarm with Tools

### Step-by-Step Guide

1. Define your tool dictionary:
```python
tool_dictionary = {
    "type": "function",
    "function": {
        "name": "search_topic",
        "description": "Conduct an in-depth search on a specified topic",
        "parameters": {
            "type": "object",
            "properties": {
                "depth": {
                    "type": "integer",
                    "description": "Search depth (1-3)"
                },
                "detailed_queries": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "Specific search queries"
                    }
                }
            },
            "required": ["depth", "detailed_queries"]
        }
    }
}
```

2. Create agent configurations:
```python
agent_config = {
    "agent_name": "Market Analyst",
    "description": "Analyzes market trends",
    "system_prompt": "You are a financial analyst expert.",
    "model_name": "openai/gpt-4",
    "role": "worker",
    "max_loops": 1,
    "max_tokens": 8192,
    "temperature": 0.5,
    "auto_generate_prompt": False,
    "tools_dictionary": [tool_dictionary]  # Optional: Add tools if needed
}
```

3. Create the swarm payload:
```python
payload = {
    "name": "Your Swarm Name",
    "description": "Swarm description",
    "agents": [agent_config],
    "max_loops": 1,
    "swarm_type": "ConcurrentWorkflow",
    "task": "Your task description",
    "output_type": "dict"
}
```

4. Make the API request:
```python
def run_swarm(payload):
    response = requests.post(
        f"{BASE_URL}/v1/swarm/completions",
        headers=headers,
        json=payload
    )
    return response.json()
```

## FAQ

### Do all agents need tools?
No, tools are optional for each agent. You can choose which agents have tools based on your specific needs. Simply omit the `tools_dictionary` field for agents that don't require tools.

### What types of tools can I use?
Currently, the API supports function-type tools. Each tool must have:
- A unique name
- A clear description
- Well-defined parameters with types and descriptions

### Can I mix agents with and without tools?
Yes, you can create swarms with a mix of tool-enabled and regular agents. This allows for flexible swarm architectures.

### What's the recommended number of tools per agent?
While there's no strict limit, it's recommended to:
- Keep tools focused and specific
- Only include tools that the agent needs
- Consider the complexity of tool interactions

## Example Implementation

Here's a complete example of a financial analysis swarm:

```python
def run_financial_analysis_swarm():
    payload = {
        "name": "Financial Analysis Swarm",
        "description": "Market analysis swarm",
        "agents": [
            {
                "agent_name": "Market Analyst",
                "description": "Analyzes market trends",
                "system_prompt": "You are a financial analyst expert.",
                "model_name": "openai/gpt-4",
                "role": "worker",
                "max_loops": 1,
                "max_tokens": 8192,
                "temperature": 0.5,
                "auto_generate_prompt": False,
                "tools_dictionary": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search_topic",
                            "description": "Conduct market research",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "depth": {
                                        "type": "integer",
                                        "description": "Search depth (1-3)"
                                    },
                                    "detailed_queries": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                },
                                "required": ["depth", "detailed_queries"]
                            }
                        }
                    }
                ]
            }
        ],
        "max_loops": 1,
        "swarm_type": "ConcurrentWorkflow",
        "task": "Analyze top performing tech ETFs",
        "output_type": "dict"
    }
    
    response = requests.post(
        f"{BASE_URL}/v1/swarm/completions",
        headers=headers,
        json=payload
    )
    return response.json()
```

## Health Check

Always verify the API status before running swarms:

```python
def check_api_health():
    response = requests.get(f"{BASE_URL}/health", headers=headers)
    return response.json()
```

## Best Practices

1. **Error Handling**: Always implement proper error handling:
```python
def safe_run_swarm(payload):
    try:
        response = requests.post(
            f"{BASE_URL}/v1/swarm/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error running swarm: {e}")
        return None
```

2. **Environment Variables**: Never hardcode API keys
3. **Tool Design**: Keep tools simple and focused
4. **Testing**: Validate swarm configurations before production use

## Troubleshooting

Common issues and solutions:

1. **API Key Issues**
   - Verify key is correctly set in `.env`
   - Check key permissions

2. **Tool Execution Errors**
   - Validate tool parameters
   - Check tool function signatures

3. **Response Timeout**
   - Consider reducing max_tokens
   - Simplify tool complexity



```python
import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

API_KEY = os.getenv("SWARMS_API_KEY")
BASE_URL = "https://api.swarms.world"

headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}


def run_health_check():
    response = requests.get(f"{BASE_URL}/health", headers=headers)
    return response.json()


def run_single_swarm():
    payload = {
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
                "auto_generate_prompt": False,
                "tools_dictionary": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search_topic",
                            "description": "Conduct an in-depth search on a specified topic or subtopic, generating a comprehensive array of highly detailed search queries tailored to the input parameters.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "depth": {
                                        "type": "integer",
                                        "description": "Indicates the level of thoroughness for the search. Values range from 1 to 3, where 1 represents a superficial search and 3 signifies an exploration of the topic.",
                                    },
                                    "detailed_queries": {
                                        "type": "array",
                                        "description": "An array of highly specific search queries that are generated based on the input query and the specified depth. Each query should be designed to elicit detailed and relevant information from various sources.",
                                        "items": {
                                            "type": "string",
                                            "description": "Each item in this array should represent a unique search query that targets a specific aspect of the main topic, ensuring a comprehensive exploration of the subject matter.",
                                        },
                                    },
                                },
                                "required": ["depth", "detailed_queries"],
                            },
                        },
                    },
                ],
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
                "auto_generate_prompt": False,
                "tools_dictionary": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search_topic",
                            "description": "Conduct an in-depth search on a specified topic or subtopic, generating a comprehensive array of highly detailed search queries tailored to the input parameters.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "depth": {
                                        "type": "integer",
                                        "description": "Indicates the level of thoroughness for the search. Values range from 1 to 3, where 1 represents a superficial search and 3 signifies an exploration of the topic.",
                                    },
                                    "detailed_queries": {
                                        "type": "array",
                                        "description": "An array of highly specific search queries that are generated based on the input query and the specified depth. Each query should be designed to elicit detailed and relevant information from various sources.",
                                        "items": {
                                            "type": "string",
                                            "description": "Each item in this array should represent a unique search query that targets a specific aspect of the main topic, ensuring a comprehensive exploration of the subject matter.",
                                        },
                                    },
                                },
                                "required": ["depth", "detailed_queries"],
                            },
                        },
                    },
                ],
            },
        ],
        "max_loops": 1,
        "swarm_type": "ConcurrentWorkflow",
        "task": "What are the best etfs and index funds for ai and tech?",
        "output_type": "dict",
    }

    response = requests.post(
        f"{BASE_URL}/v1/swarm/completions",
        headers=headers,
        json=payload,
    )

    print(response)
    print(response.status_code)
    # return response.json()
    output = response.json()

    return json.dumps(output, indent=4)


if __name__ == "__main__":
    result = run_single_swarm()
    print("Swarm Result:")
    print(result)

```