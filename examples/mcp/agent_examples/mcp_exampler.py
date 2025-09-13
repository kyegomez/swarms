from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)

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
                        "description": "Indicates whether to include historical price data along with the current price.",
                    },
                    "time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Optional parameter to specify the time for which the stock data is requested, in ISO 8601 format.",
                    },
                },
                "required": [
                    "ticker",
                    "include_history",
                    "time",
                ],
            },
        },
    }
]


# Initialize the agent
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    agent_description="Personal finance advisor agent",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    max_loops=1,
    # tools_list_dictionary=tools,
    # mcp_servers=["http://localhost:8000/mcp"],
    # output_type="dict-all-except-first",
    #     "dict-final",
    # "dict-all-except-first",
    # "str-all-except-first",
)

print(
    agent.run(
        "What is the current stock price for Apple Inc. (AAPL)? Include historical price data.",
    )
)
