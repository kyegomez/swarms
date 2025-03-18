# Agentic Structured Outputs

Structured outputs help ensure that your agents return data in a consistent, predictable format that can be easily parsed and processed by your application. This is particularly useful when building complex applications that require standardized data handling.

## Schema Definition

Structured outputs are defined using JSON Schema format. Here's the basic structure:

```python
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

### Parameter Types

The following parameter types are supported:

- `string`: Text values
- `number`: Numeric values
- `boolean`: True/False values
- `object`: Nested objects
- `array`: Lists or arrays
- `null`: Null values

## Implementation Steps

1. **Define Your Schema**
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
                       # Add more parameters as needed
                   },
                   "required": ["ticker"]
               }
           }
       }
   ]
   ```

2. **Initialize the Agent**
   ```python
   from swarms import Agent
   
   agent = Agent(
       agent_name="Your-Agent-Name",
       agent_description="Agent description",
       system_prompt="Your system prompt",
       tools_list_dictionary=tools
   )
   ```

3. **Run the Agent**
   ```python
   response = agent.run("Your query here")
   ```

4. **Parse the Output**
   ```python
   from swarms.utils.str_to_dict import str_to_dict
   
   parsed_output = str_to_dict(response)
   ```

## Example Usage

Here's a complete example using a financial agent:

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
                        "description": "Stock ticker symbol"
                    },
                    "include_history": {
                        "type": "boolean",
                        "description": "Include historical data"
                    },
                    "time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Time for stock data"
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
    system_prompt="Your system prompt here",
    max_loops=1,
    tools_list_dictionary=tools
)

# Run agent
response = agent.run("What is the current stock price for AAPL?")

# Parse structured output
parsed_data = str_to_dict(response)
```

## Best Practices

1. **Schema Design**
   - Keep schemas as simple as possible while meeting your needs
   - Use clear, descriptive parameter names
   - Include detailed descriptions for each parameter
   - Specify all required parameters explicitly

2. **Error Handling**
   - Always validate the output format
   - Implement proper error handling for parsing failures
   - Use try-except blocks when converting strings to dictionaries

3. **Performance**
   - Minimize the number of required parameters
   - Use appropriate data types for each parameter
   - Consider caching parsed results if used frequently

## Troubleshooting

Common issues and solutions:

1. **Invalid Output Format**
   - Ensure your schema matches the expected output
   - Verify all required fields are present
   - Check for proper JSON formatting

2. **Parsing Errors**
   - Use `str_to_dict()` for reliable string-to-dictionary conversion
   - Validate input strings before parsing
   - Handle potential parsing exceptions

3. **Missing Fields**
   - Verify all required fields are defined in the schema
   - Check if the agent is properly configured
   - Review the system prompt for clarity

