"""
* WORKING
What this script does:
Structured output example with validation function
Requirements:
pip install openai
pip install pydantic
Add the folowing API key(s) in your .env file:
   - OPENAI_API_KEY (this example works best with Openai bc it uses openai function calling structure)
"""

################ Adding project root to PYTHONPATH ################################
# If you are running playground examples in the project files directly, use this: 

import sys
import os

sys.path.insert(0, os.getcwd())

################ Adding project root to PYTHONPATH ################################

from swarms import Agent, OpenAIChat

from pydantic import BaseModel, Field
from typing_extensions import Annotated
from pydantic import AfterValidator


def symbol_must_exists(symbol= str) -> str:
    symbols = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "NVDA", "BRK.B", 
    "JPM", "JNJ", "V", "PG", "UNH", "MA", "HD", "BAC", "XOM", "DIS", "CSCO"
    ]
    if symbol not in symbols:
        raise ValueError(f"symbol must exists in the list: {symbols}")

    return symbol


# Initialize the schema for the person's information
class StockInfo(BaseModel):
    """
    To create a StockInfo, you need to return a JSON object with the following format:
    {
        "function_call": "StockInfo",
        "parameters": {
            ...
        }
    } 
    """
    name: str = Field(..., title="Name of the company")
    description: str = Field(..., title="Description of the company")
    symbol: Annotated[str, AfterValidator(symbol_must_exists)] = Field(..., title="stock symbol of the company")


# Define the task to generate a person's information
task = "Generate an existing S&P500's company information"

# Initialize the agent
agent = Agent(
    agent_name="Stock Information Generator",
    system_prompt=(
        "Generate a public comapany's information"
    ),
    llm=OpenAIChat(),
    max_loops=1,
    verbose=True,
    # List of schemas that the agent can handle
    list_base_models=[StockInfo],
    output_validation=True,
)

# Run the agent to generate the person's information
generated_data = agent.run(task)

# Print the generated data
print(f"Generated data: {generated_data}")