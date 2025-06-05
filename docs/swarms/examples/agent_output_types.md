# Agent Output Types Examples

This example demonstrates how to use different output types when working with Swarms agents. Each output type formats the agent's response in a specific way, making it easier to integrate with different parts of your application.

## Prerequisites

- Python 3.7+
- OpenAI API key
- Swarms library

## Installation

```bash
pip3 install -U swarms
```

## Environment Variables

```plaintext
WORKSPACE_DIR="agent_workspace"
OPENAI_API_KEY=""
ANTHROPIC_API_KEY=""
```

## Available Output Types

The following output types are supported:

| Output Type | Description |
|------------|-------------|
| `"list"` | Returns response as a JSON string containing a list |
| `"dict"` or `"dictionary"` | Returns response as a Python dictionary |
| `"string"` or `"str"` | Returns response as a plain string |
| `"final"` or `"last"` | Returns only the final response |
| `"json"` | Returns response as a JSON string |
| `"all"` | Returns all responses in the conversation |
| `"yaml"` | Returns response formatted as YAML |
| `"xml"` | Returns response formatted as XML |
| `"dict-all-except-first"` | Returns all responses except the first as a dictionary |
| `"str-all-except-first"` | Returns all responses except the first as a string |
| `"basemodel"` | Returns response as a Pydantic BaseModel |

## Examples

### 1. String Output (Default)

```python
from swarms import Agent

# Initialize agent with string output
agent = Agent(
    agent_name="String-Output-Agent",
    agent_description="Demonstrates string output format",
    system_prompt="You are a helpful assistant that provides clear text responses.",
    output_type="str",  # or "string"
)

response = agent.run("What is the capital of France?")

```

### 2. JSON Output

```python
# Initialize agent with JSON output
agent = Agent(
    agent_name="JSON-Output-Agent",
    agent_description="Demonstrates JSON output format",
    system_prompt="You are an assistant that provides structured data responses.",
    output_type="json"
)

response = agent.run("List the top 3 programming languages.")

```

### 3. List Output

```python
# Initialize agent with list output
agent = Agent(
    agent_name="List-Output-Agent",
    agent_description="Demonstrates list output format",
    system_prompt="You are an assistant that provides list-based responses.",
    output_type="list"
)

response = agent.run("Name three primary colors.")

```

### 4. Dictionary Output

```python
# Initialize agent with dictionary output
agent = Agent(
    agent_name="Dict-Output-Agent",
    agent_description="Demonstrates dictionary output format",
    system_prompt="You are an assistant that provides dictionary-based responses.",
    output_type="dict"  # or "dictionary"
)

response = agent.run("Provide information about a book.")

```

### 5. YAML Output

```python
# Initialize agent with YAML output
agent = Agent(
    agent_name="YAML-Output-Agent",
    agent_description="Demonstrates YAML output format",
    system_prompt="You are an assistant that provides YAML-formatted responses.",
    output_type="yaml"
)

response = agent.run("Describe a recipe.")
```

### 6. XML Output

```python
# Initialize agent with XML output
agent = Agent(
    agent_name="XML-Output-Agent",
    agent_description="Demonstrates XML output format",
    system_prompt="You are an assistant that provides XML-formatted responses.",
    output_type="xml"
)

response = agent.run("Provide user information.")
```

### 7. All Responses

```python
# Initialize agent to get all responses
agent = Agent(
    agent_name="All-Output-Agent",
    agent_description="Demonstrates getting all responses",
    system_prompt="You are an assistant that provides multiple responses.",
    output_type="all"
)

response = agent.run("Tell me about climate change.")
```

### 8. Final Response Only

```python
# Initialize agent to get only final response
agent = Agent(
    agent_name="Final-Output-Agent",
    agent_description="Demonstrates getting only final response",
    system_prompt="You are an assistant that provides concise final answers.",
    output_type="final"  # or "last"
)

response = agent.run("What's the meaning of life?")
```


## Best Practices

1. Choose the output type based on your application's needs:
   
   | Output Type | Use Case |
   |------------|----------|
   | `"str"` | Simple text responses |
   | `"json"` or `"dict"` | Structured data |
   | `"list"` | Array-like data |
   | `"yaml"` | Configuration-like data |
   | `"xml"` | XML-based integrations |
   | `"basemodel"` | Type-safe data handling |

2. Handle the output appropriately in your application:

   - Parse JSON/YAML responses when needed
   
   - Validate structured data
   
   - Handle potential formatting errors

3. Consider using `try-except` blocks when working with structured output types to handle potential parsing errors.


This comprehensive guide shows how to use all available output types in the Swarms framework, making it easier to integrate agent responses into your applications in the most suitable format for your needs.