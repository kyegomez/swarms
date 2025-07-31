"""
Stagehand Tools for Swarms Agent
=================================

This example demonstrates how to create Stagehand browser automation tools
that can be used by a standard Swarms Agent. Each Stagehand method (act, 
extract, observe) becomes a separate tool that the agent can use.

This approach gives the agent more fine-grained control over browser
automation tasks.
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field

from swarms import Agent
from swarms.tools.base_tool import BaseTool
from stagehand import Stagehand, StagehandConfig

load_dotenv()


class BrowserState:
    """Singleton to manage browser state across tools."""

    _instance = None
    _stagehand = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def init_browser(
        self,
        env: str = "LOCAL",
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        model_api_key: Optional[str] = None,
    ):
        """Initialize the browser if not already initialized."""
        if not self._initialized:
            config = StagehandConfig(
                env=env,
                api_key=api_key or os.getenv("BROWSERBASE_API_KEY"),
                project_id=project_id or os.getenv("BROWSERBASE_PROJECT_ID"),
                model_name=model_name,
                model_api_key=model_api_key or os.getenv("OPENAI_API_KEY"),
            )
            self._stagehand = Stagehand(config)
            await self._stagehand.init()
            self._initialized = True
            logger.info("Stagehand browser initialized")

    async def get_page(self):
        """Get the current page instance."""
        if not self._initialized:
            raise RuntimeError("Browser not initialized. Call init_browser first.")
        return self._stagehand.page

    async def close(self):
        """Close the browser."""
        if self._initialized and self._stagehand:
            await self._stagehand.close()
            self._initialized = False
            logger.info("Stagehand browser closed")


# Browser state instance
browser_state = BrowserState()


class NavigateTool(BaseTool):
    """Tool for navigating to URLs in the browser."""

    def __init__(self):
        super().__init__(
            name="navigate_browser",
            description="Navigate to a URL in the browser. Input should be a valid URL starting with http:// or https://",
            verbose=True,
        )

    def run(self, url: str) -> str:
        """Navigate to the specified URL."""
        return asyncio.run(self._async_run(url))

    async def _async_run(self, url: str) -> str:
        try:
            await browser_state.init_browser()
            page = await browser_state.get_page()
            
            # Ensure URL has protocol
            if not url.startswith(("http://", "https://")):
                url = f"https://{url}"
            
            await page.goto(url)
            return f"Successfully navigated to {url}"
        except Exception as e:
            logger.error(f"Navigation error: {str(e)}")
            return f"Failed to navigate to {url}: {str(e)}"


class ActTool(BaseTool):
    """Tool for performing actions on web pages."""

    def __init__(self):
        super().__init__(
            name="browser_act",
            description=(
                "Perform an action on the current web page using natural language. "
                "Examples: 'click the submit button', 'type hello@example.com in the email field', "
                "'scroll down', 'press Enter'"
            ),
            verbose=True,
        )

    def run(self, action: str) -> str:
        """Perform the specified action."""
        return asyncio.run(self._async_run(action))

    async def _async_run(self, action: str) -> str:
        try:
            await browser_state.init_browser()
            page = await browser_state.get_page()
            
            result = await page.act(action)
            return f"Action performed: {action}. Result: {result}"
        except Exception as e:
            logger.error(f"Action error: {str(e)}")
            return f"Failed to perform action '{action}': {str(e)}"


class ExtractTool(BaseTool):
    """Tool for extracting data from web pages."""

    def __init__(self):
        super().__init__(
            name="browser_extract",
            description=(
                "Extract information from the current web page using natural language. "
                "Examples: 'extract all email addresses', 'get the main article text', "
                "'find all product prices', 'extract the page title and meta description'"
            ),
            verbose=True,
        )

    def run(self, query: str) -> str:
        """Extract information based on the query."""
        return asyncio.run(self._async_run(query))

    async def _async_run(self, query: str) -> str:
        try:
            await browser_state.init_browser()
            page = await browser_state.get_page()
            
            extracted = await page.extract(query)
            
            # Convert to JSON string for agent consumption
            if isinstance(extracted, (dict, list)):
                return json.dumps(extracted, indent=2)
            else:
                return str(extracted)
        except Exception as e:
            logger.error(f"Extraction error: {str(e)}")
            return f"Failed to extract '{query}': {str(e)}"


class ObserveTool(BaseTool):
    """Tool for observing elements on web pages."""

    def __init__(self):
        super().__init__(
            name="browser_observe",
            description=(
                "Observe and find elements on the current web page using natural language. "
                "Returns information about elements including their selectors. "
                "Examples: 'find the search box', 'locate the submit button', "
                "'find all navigation links'"
            ),
            verbose=True,
        )

    def run(self, query: str) -> str:
        """Observe elements based on the query."""
        return asyncio.run(self._async_run(query))

    async def _async_run(self, query: str) -> str:
        try:
            await browser_state.init_browser()
            page = await browser_state.get_page()
            
            observations = await page.observe(query)
            
            # Format observations for readability
            result = []
            for obs in observations:
                result.append({
                    "description": obs.description,
                    "selector": obs.selector,
                    "method": obs.method
                })
            
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Observation error: {str(e)}")
            return f"Failed to observe '{query}': {str(e)}"


class ScreenshotTool(BaseTool):
    """Tool for taking screenshots of the current page."""

    def __init__(self):
        super().__init__(
            name="browser_screenshot",
            description="Take a screenshot of the current web page. Optionally provide a filename.",
            verbose=True,
        )

    def run(self, filename: str = "screenshot.png") -> str:
        """Take a screenshot."""
        return asyncio.run(self._async_run(filename))

    async def _async_run(self, filename: str) -> str:
        try:
            await browser_state.init_browser()
            page = await browser_state.get_page()
            
            # Ensure .png extension
            if not filename.endswith(".png"):
                filename += ".png"
            
            # Get the underlying Playwright page
            playwright_page = page.page
            await playwright_page.screenshot(path=filename)
            
            return f"Screenshot saved to {filename}"
        except Exception as e:
            logger.error(f"Screenshot error: {str(e)}")
            return f"Failed to take screenshot: {str(e)}"


class CloseBrowserTool(BaseTool):
    """Tool for closing the browser."""

    def __init__(self):
        super().__init__(
            name="close_browser",
            description="Close the browser when done with automation tasks",
            verbose=True,
        )

    def run(self, *args) -> str:
        """Close the browser."""
        return asyncio.run(self._async_run())

    async def _async_run(self) -> str:
        try:
            await browser_state.close()
            return "Browser closed successfully"
        except Exception as e:
            logger.error(f"Close browser error: {str(e)}")
            return f"Failed to close browser: {str(e)}"


# Example usage
if __name__ == "__main__":
    # Create browser automation tools
    navigate_tool = NavigateTool()
    act_tool = ActTool()
    extract_tool = ExtractTool()
    observe_tool = ObserveTool()
    screenshot_tool = ScreenshotTool()
    close_browser_tool = CloseBrowserTool()

    # Create a Swarms agent with browser tools
    browser_agent = Agent(
        agent_name="BrowserAutomationAgent",
        model_name="gpt-4o-mini",
        max_loops=1,
        tools=[
            navigate_tool,
            act_tool,
            extract_tool,
            observe_tool,
            screenshot_tool,
            close_browser_tool,
        ],
        system_prompt="""You are a web browser automation specialist. You can:
        1. Navigate to websites using the navigate_browser tool
        2. Perform actions like clicking and typing using the browser_act tool
        3. Extract information from pages using the browser_extract tool
        4. Find and observe elements using the browser_observe tool
        5. Take screenshots using the browser_screenshot tool
        6. Close the browser when done using the close_browser tool

        Always start by navigating to a URL before trying to interact with a page.
        Be specific in your actions and extractions. When done with tasks, close the browser.""",
    )

    # Example 1: Research task
    print("Example 1: Automated web research")
    result1 = browser_agent.run(
        "Go to hackernews (news.ycombinator.com) and extract the titles of the top 5 stories. Then take a screenshot."
    )
    print(result1)
    print("\n" + "=" * 50 + "\n")

    # Example 2: Search task
    print("Example 2: Perform a web search")
    result2 = browser_agent.run(
        "Navigate to google.com, search for 'Python web scraping best practices', and extract the first 3 search result titles"
    )
    print(result2)
    print("\n" + "=" * 50 + "\n")

    # Example 3: Form interaction
    print("Example 3: Interact with a form")
    result3 = browser_agent.run(
        "Go to example.com and observe what elements are on the page. Then extract all the text content."
    )
    print(result3)

    # Clean up
    browser_agent.run("Close the browser")