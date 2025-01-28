from swarms.tools.base_tool import BaseTool

import requests
from swarms.utils.litellm_wrapper import LiteLLM


def get_stock_data(symbol: str) -> str:
    """
    Fetches stock data from Yahoo Finance for a given stock symbol.

    Args:
        symbol (str): The stock symbol to fetch data for (e.g., 'AAPL' for Apple Inc.).

    Returns:
        Dict[str, Any]: A dictionary containing stock data, including price, volume, and other relevant information.

    Raises:
        ValueError: If the stock symbol is invalid or data cannot be retrieved.
    """
    url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}"
    response = requests.get(url)

    if response.status_code != 200:
        raise ValueError(f"Error fetching data for symbol: {symbol}")

    data = response.json()
    if (
        "quoteResponse" not in data
        or not data["quoteResponse"]["result"]
    ):
        raise ValueError(f"No data found for symbol: {symbol}")

    return str(data["quoteResponse"]["result"][0])


tool_schema = BaseTool(
    tools=[get_stock_data]
).convert_tool_into_openai_schema()

tool_schema = tool_schema["functions"][0]

llm = LiteLLM(
    model_name="gpt-4o",
)

print(
    llm.run(
        "What is the stock data for Apple Inc. (AAPL)?",
        tools=[tool_schema],
    )
)
