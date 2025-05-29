# :material-code-json: Agentic Structured Outputs

!!! abstract "Overview"
    Structured outputs help ensure that your agents return data in a consistent, predictable format that can be easily parsed and processed by your application. This is particularly useful when building complex applications that require standardized data handling.

## :material-file-document-outline: Schema Definition

Structured outputs are defined using JSON Schema format. Here's the basic structure:

=== "Basic Schema"

    ```python title="Basic Tool Schema"
    tools = [
        {
            "type": "function",
            "function": {
                "name": "function_name",
                "description": "Description of what the function does",
                "parameters": {
                    "type": "object",
                    "properties": {
                        # Define your parameters here
                    },
                    "required": [
                        # List required parameters
                    ]
                }
            }
        }
    ]
    ```

=== "Advanced Schema"

    ```python title="Advanced Tool Schema with Multiple Parameters"
    tools = [
        {
            "type": "function",
            "function": {
                "name": "advanced_function",
                "description": "Advanced function with multiple parameter types",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text_param": {
                            "type": "string",
                            "description": "A text parameter"
                        },
                        "number_param": {
                            "type": "number",
                            "description": "A numeric parameter"
                        },
                        "boolean_param": {
                            "type": "boolean",
                            "description": "A boolean parameter"
                        },
                        "array_param": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "An array of strings"
                        }
                    },
                    "required": ["text_param", "number_param"]
                }
            }
        }
    ]
    ```

### :material-format-list-bulleted-type: Parameter Types

The following parameter types are supported:

| Type | Description | Example |
|------|-------------|---------|
| `string` | Text values | `"Hello World"` |
| `number` | Numeric values | `42`, `3.14` |
| `boolean` | True/False values | `true`, `false` |
| `object` | Nested objects | `{"key": "value"}` |
| `array` | Lists or arrays | `[1, 2, 3]` |
| `null` | Null values | `null` |

## :material-cog: Implementation Steps

!!! tip "Quick Start Guide"
    Follow these steps to implement structured outputs in your agent:

### Step 1: Define Your Schema

```python 
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Retrieve stock price information",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    },
                    "include_volume": {
                        "type": "boolean",
                        "description": "Include trading volume data"
                    }
                },
                "required": ["ticker"]
            }
        }
    }
]
```

### Step 2: Initialize the Agent

```
from swarms import Agent

agent = Agent(
    agent_name="Your-Agent-Name",
    agent_description="Agent description",
    system_prompt="Your system prompt",
    tools_list_dictionary=tools
)
```

### Step 3: Run the Agent

```python
response = agent.run("Your query here")
```

### Step 4: Parse the Output

```python
from swarms.utils.str_to_dict import str_to_dict

parsed_output = str_to_dict(response)
```

## :material-code-braces: Example Usage

!!! example "Complete Financial Agent Example"
    Here's a comprehensive example using a financial analysis agent:

=== "Python Implementation"

    ```python
    from dotenv import load_dotenv
    from swarms import Agent
    from swarms.utils.str_to_dict import str_to_dict
    
    # Load environment variables
    load_dotenv()
    
    # Define tools with structured output schema
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "Retrieve the current stock price and related information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "Stock ticker symbol (e.g., AAPL, GOOGL)"
                        },
                        "include_history": {
                            "type": "boolean",
                            "description": "Include historical data in the response"
                        },
                        "time": {
                            "type": "string",
                            "format": "date-time",
                            "description": "Specific time for stock data (ISO format)"
                        }
                    },
                    "required": ["ticker", "include_history", "time"]
                }
            }
        }
    ]
    
    # Initialize agent
    agent = Agent(
        agent_name="Financial-Analysis-Agent",
        agent_description="Personal finance advisor agent",
        system_prompt="You are a helpful financial analysis assistant.",
        max_loops=1,
        tools_list_dictionary=tools
    )
    
    # Run agent
    response = agent.run("What is the current stock price for AAPL?")
    
    # Parse structured output
    parsed_data = str_to_dict(response)
    print(f"Parsed response: {parsed_data}")
    ```

=== "Expected Output"

    ```json
    {
        "function_calls": [
            {
                "name": "get_stock_price",
                "arguments": {
                    "ticker": "AAPL",
                    "include_history": true,
                    "time": "2024-01-15T10:30:00Z"
                }
            }
        ]
    }
    ```

## :material-check-circle: Best Practices

!!! success "Schema Design"
    
    - **Keep it simple**: Design schemas that are as simple as possible while meeting your needs
    
    - **Clear naming**: Use descriptive parameter names that clearly indicate their purpose
    
    - **Detailed descriptions**: Include comprehensive descriptions for each parameter
    
    - **Required fields**: Explicitly specify all required parameters

!!! info "Error Handling"
    
    - **Validate output**: Always validate the output format before processing
    
    - **Exception handling**: Implement proper error handling for parsing failures
    
    - **Safety first**: Use try-except blocks when converting strings to dictionaries

!!! performance "Performance Tips"
    
    - **Minimize requirements**: Keep the number of required parameters to a minimum
    
    - **Appropriate types**: Use the most appropriate data types for each parameter
    
    - **Caching**: Consider caching parsed results if they're used frequently

## :material-alert-circle: Troubleshooting

!!! warning "Common Issues"

### Invalid Output Format

!!! failure "Problem"
    The agent returns data in an unexpected format

!!! success "Solution"
    
    - Ensure your schema matches the expected output structure
    
    - Verify all required fields are present in the response
    
    - Check for proper JSON formatting in the output

### Parsing Errors

!!! failure "Problem"
    Errors occur when trying to parse the agent's response

!!! success "Solution"
    
    ```python
    from swarms.utils.str_to_dict import str_to_dict
    
    try:
        parsed_data = str_to_dict(response)
    except Exception as e:
        print(f"Parsing error: {e}")
        # Handle the error appropriately
    ```

### Missing Fields

!!! failure "Problem"
    Required fields are missing from the output

!!! success "Solution"
    - Verify all required fields are defined in the schema
    - Check if the agent is properly configured with the tools
    - Review the system prompt for clarity and completeness

## :material-lightbulb: Advanced Features

!!! note "Pro Tips"
    
    === "Nested Objects"
    
        ```python title="nested_schema.py"
        "properties": {
            "user_info": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "number"},
                    "preferences": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }
        }
        ```
    
    === "Conditional Fields"
    
        ```python title="conditional_schema.py"
        "properties": {
            "data_type": {
                "type": "string",
                "enum": ["stock", "crypto", "forex"]
            },
            "symbol": {"type": "string"},
            "exchange": {
                "type": "string",
                "description": "Required for crypto and forex"
            }
        }
        ```

---

