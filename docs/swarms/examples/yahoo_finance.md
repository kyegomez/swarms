# Yahoo Finance Integration with Swarms


This example demonstrates how to integrate Yahoo Finance data into your Swarms agents using the `swarms-tools` package. The agent can analyze real-time financial data, stock metrics, and market information by making function calls to the Yahoo Finance API. This is particularly useful for financial analysis, portfolio management, and market research applications.

## Install

```bash
pip3 install -U swarms swarms-tools
```

## Environment Variables

```txt
# OpenAI API Key (Required for LLM functionality)
OPENAI_API_KEY="your_openai_api_key_here"
```

## Usage

1. Install the required packages
2. Add your `OPENAI_API_KEY` to your `.env` file
3. Run the example code below
4. The agent will make a function call to the Yahoo Finance tool
5. The tool will execute and return financial data
6. The agent analyzes the result and provides insights

## Code Example

```python
from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)
from swarms_tools import (
    yahoo_finance_api,
)

# Initialize the agent
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    agent_description="Personal finance advisor agent",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    max_loops=1,
    model_name="gpt-4.1",
    tools=[yahoo_finance_api],
)

# Run financial analysis
agent.run("Analyze the latest metrics for nvidia")
```

**Result**: Less than 30 lines of code to get a fully functional financial analysis agent!

