# Agent Structured Outputs

This example demonstrates how to use structured outputs with Swarms agents following OpenAI's function calling schema. By defining function schemas, you can specify exactly how agents should structure their responses, making it easier to parse and use the outputs in your applications.

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

## Understanding Function Schemas

Function schemas in Swarms follow OpenAI's function calling format. Each function schema is defined as a dictionary with the following structure:

```python
{
    "type": "function",
    "function": {
        "name": "function_name",
        "description": "A clear description of what the function does",
        "parameters": {
            "type": "object",
            "properties": {
                # Define the parameters your function accepts
            },
            "required": ["list", "of", "required", "parameters"]
        }
    }
}
```

## Code Example

Here's an example showing how to use multiple function schemas with a Swarms agent:

```python
from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import FINANCIAL_AGENT_SYS_PROMPT

# Define multiple function schemas
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Retrieve the current stock price and related information for a specified company.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol of the company, e.g. AAPL for Apple Inc.",
                    },
                    "include_history": {
                        "type": "boolean",
                        "description": "Whether to include historical price data.",
                    },
                    "time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Optional time for stock data, in ISO 8601 format.",
                    },
                },
                "required": ["ticker", "include_history"]
            },
        },
    },
    # Can pass in multiple function schemas as well
    {
        "type": "function",
        "function": {
            "name": "analyze_company_financials",
            "description": "Analyze key financial metrics and ratios for a company.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol of the company",
                    },
                    "metrics": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["PE_ratio", "market_cap", "revenue", "profit_margin"]
                        },
                        "description": "List of financial metrics to analyze"
                    },
                    "timeframe": {
                        "type": "string",
                        "enum": ["quarterly", "annual", "ttm"],
                        "description": "Timeframe for the analysis"
                    }
                },
                "required": ["ticker", "metrics"]
            }
        }
    }
]

# Initialize the agent with multiple function schemas
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    agent_description="Personal finance advisor agent that can fetch stock prices and analyze financials",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    max_loops=1,
    tools_list_dictionary=tools,  # Pass in the list of function schemas
    output_type="final"
)

# Example usage with stock price query
stock_response = agent.run(
    "What is the current stock price for Apple Inc. (AAPL)? Include historical data."
)
print("Stock Price Response:", stock_response)

# Example usage with financial analysis query
analysis_response = agent.run(
    "Analyze Apple's PE ratio and market cap using quarterly data."
)
print("Financial Analysis Response:", analysis_response)
```

## Schema Types and Properties

The function schema supports various parameter types and properties:

| Schema Type | Description |
|------------|-------------|
| Basic Types | `string`, `number`, `integer`, `boolean`, `array`, `object` |
| Format Specifications | `date-time`, `date`, `email`, etc. |
| Enums | Restrict values to a predefined set |
| Required vs Optional Parameters | Specify which parameters must be provided |
| Nested Objects and Arrays | Support for complex data structures |

Example of a more complex schema:

```python
{
    "type": "function",
    "function": {
        "name": "generate_investment_report",
        "description": "Generate a comprehensive investment report",
        "parameters": {
            "type": "object",
            "properties": {
                "portfolio": {
                    "type": "object",
                    "properties": {
                        "stocks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "ticker": {"type": "string"},
                                    "shares": {"type": "number"},
                                    "entry_price": {"type": "number"}
                                }
                            }
                        },
                        "risk_tolerance": {
                            "type": "string",
                            "enum": ["low", "medium", "high"]
                        },
                        "time_horizon": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 30,
                            "description": "Investment time horizon in years"
                        }
                    },
                    "required": ["stocks", "risk_tolerance"]
                },
                "report_type": {
                    "type": "string",
                    "enum": ["summary", "detailed", "risk_analysis"]
                }
            },
            "required": ["portfolio"]
        }
    }
}
```

This example shows how to structure complex nested objects, arrays, and various parameter types while following OpenAI's function calling schema.
