# Browser Automation with Swarms

This example demonstrates how to use browser automation capabilities within the Swarms framework. The `BrowserUseAgent` class provides a powerful interface for web scraping, navigation, and automated browser interactions using the `browser_use` library. This is particularly useful for tasks that require real-time web data extraction, form filling, or web application testing.

## Install

```bash
pip3 install -U swarms browser-use python-dotenv langchain-openai
```

## Environment Variables

```txt
# OpenAI API Key (Required for LLM functionality)
OPENAI_API_KEY="your_openai_api_key_here"
```

## Main Code

```python 
import asyncio

from browser_use import Agent as BrowserAgent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from swarms import Agent

load_dotenv()


class BrowserUseAgent:
    def __init__(self, agent_name: str = "BrowserAgent", agent_description: str = "A browser agent that can navigate the web and perform tasks."):
        """
        Initialize a BrowserAgent with a given name.

        Args:
            agent_name (str): The name of the browser agent.
        """
        self.agent_name = agent_name
        self.agent_description = agent_description

    async def browser_agent_test(self, task: str):
        """
        Asynchronously run the browser agent on a given task.

        Args:
            task (str): The task prompt for the agent.

        Returns:
            Any: The result of the agent's run method.
        """
        agent = BrowserAgent(
            task=task,
            llm=ChatOpenAI(model="gpt-4.1"),
        )
        result = await agent.run()
        return result.model_dump_json(indent=4)

    def run(self, task: str):
        """
        Run the browser agent synchronously on a given task.

        Args:
            task (str): The task prompt for the agent.

        Returns:
            Any: The result of the agent's run method.
        """
        return asyncio.run(self.browser_agent_test(task))



def browser_agent_tool(task: str):
    """
    Executes a browser automation agent as a callable tool.

    This function instantiates a `BrowserAgent` and runs it synchronously on the provided task prompt.
    The agent will use a language model to interpret the task, control a browser, and return the results
    as a JSON-formatted string.

    Args:
        task (str): 
            A detailed instruction or prompt describing the browser-based task to perform.
            For example, you can instruct the agent to navigate to a website, extract information,
            or interact with web elements.

    Returns:
        str:
            The result of the browser agent's execution, formatted as a JSON string. The output
            typically includes the agent's findings, extracted data, and any relevant observations
            from the automated browser session.

    Example:
        result = browser_agent_tool(
            "Please navigate to https://www.coingecko.com and identify the best performing cryptocurrency coin over the past 24 hours."
        )
        print(result)
    """
    return BrowserAgent().run(task)



agent = Agent(
    name = "Browser Agent",
    model_name = "gpt-4.1",
    tools = [browser_agent_tool],
)

agent.run("Please navigate to https://www.coingecko.com and identify the best performing cryptocurrency coin over the past 24 hours.")
```