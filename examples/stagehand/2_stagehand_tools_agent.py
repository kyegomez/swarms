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
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

from swarms import Agent
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
                project_id=project_id
                or os.getenv("BROWSERBASE_PROJECT_ID"),
                model_name=model_name,
                model_api_key=model_api_key
                or os.getenv("OPENAI_API_KEY"),
            )
            self._stagehand = Stagehand(config)
            await self._stagehand.init()
            self._initialized = True
            logger.info("Stagehand browser initialized")

    async def get_page(self):
        """Get the current page instance."""
        if not self._initialized:
            raise RuntimeError(
                "Browser not initialized. Call init_browser first."
            )
        return self._stagehand.page

    async def close(self):
        """Close the browser."""
        if self._initialized and self._stagehand:
            await self._stagehand.close()
            self._initialized = False
            logger.info("Stagehand browser closed")


# Browser state instance
browser_state = BrowserState()


def navigate_browser(url: str) -> str:
    """
    Navigate to a URL in the browser.

    Args:
        url (str): The URL to navigate to. Should be a valid URL starting with http:// or https://.
                  If no protocol is provided, https:// will be added automatically.

    Returns:
        str: Success message with the URL navigated to, or error message if navigation fails

    Raises:
        RuntimeError: If browser initialization fails
        Exception: If navigation to the URL fails

    Example:
        >>> result = navigate_browser("https://example.com")
        >>> print(result)
        "Successfully navigated to https://example.com"

        >>> result = navigate_browser("google.com")
        >>> print(result)
        "Successfully navigated to https://google.com"
    """
    return asyncio.run(_navigate_browser_async(url))


async def _navigate_browser_async(url: str) -> str:
    """Async implementation of navigate_browser."""
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


def browser_act(action: str) -> str:
    """
    Perform an action on the current web page using natural language.

    Args:
        action (str): Natural language description of the action to perform.
                     Examples: 'click the submit button', 'type hello@example.com in the email field',
                     'scroll down', 'press Enter', 'select option from dropdown'

    Returns:
        str: JSON formatted string with action result and status information

    Raises:
        RuntimeError: If browser is not initialized or page is not available
        Exception: If the action cannot be performed on the current page

    Example:
        >>> result = browser_act("click the submit button")
        >>> print(result)
        "Action performed: click the submit button. Result: clicked successfully"

        >>> result = browser_act("type hello@example.com in the email field")
        >>> print(result)
        "Action performed: type hello@example.com in the email field. Result: text entered"
    """
    return asyncio.run(_browser_act_async(action))


async def _browser_act_async(action: str) -> str:
    """Async implementation of browser_act."""
    try:
        await browser_state.init_browser()
        page = await browser_state.get_page()

        result = await page.act(action)
        return f"Action performed: {action}. Result: {result}"
    except Exception as e:
        logger.error(f"Action error: {str(e)}")
        return f"Failed to perform action '{action}': {str(e)}"


def browser_extract(query: str) -> str:
    """
    Extract information from the current web page using natural language.

    Args:
        query (str): Natural language description of what information to extract.
                    Examples: 'extract all email addresses', 'get the main article text',
                    'find all product prices', 'extract the page title and meta description'

    Returns:
        str: JSON formatted string containing the extracted information, or error message if extraction fails

    Raises:
        RuntimeError: If browser is not initialized or page is not available
        Exception: If extraction fails due to page content or parsing issues

    Example:
        >>> result = browser_extract("extract all email addresses")
        >>> print(result)
        '["contact@example.com", "support@example.com"]'

        >>> result = browser_extract("get the main article text")
        >>> print(result)
        '{"title": "Article Title", "content": "Article content..."}'
    """
    return asyncio.run(_browser_extract_async(query))


async def _browser_extract_async(query: str) -> str:
    """Async implementation of browser_extract."""
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


def browser_observe(query: str) -> str:
    """
    Observe and find elements on the current web page using natural language.

    Args:
        query (str): Natural language description of elements to find.
                    Examples: 'find the search box', 'locate the submit button',
                    'find all navigation links', 'observe form elements'

    Returns:
        str: JSON formatted string containing information about found elements including
             their descriptions, selectors, and interaction methods

    Raises:
        RuntimeError: If browser is not initialized or page is not available
        Exception: If observation fails due to page structure or element detection issues

    Example:
        >>> result = browser_observe("find the search box")
        >>> print(result)
        '[{"description": "Search input field", "selector": "#search", "method": "input"}]'

        >>> result = browser_observe("locate the submit button")
        >>> print(result)
        '[{"description": "Submit button", "selector": "button[type=submit]", "method": "click"}]'
    """
    return asyncio.run(_browser_observe_async(query))


async def _browser_observe_async(query: str) -> str:
    """Async implementation of browser_observe."""
    try:
        await browser_state.init_browser()
        page = await browser_state.get_page()

        observations = await page.observe(query)

        # Format observations for readability
        result = []
        for obs in observations:
            result.append(
                {
                    "description": obs.description,
                    "selector": obs.selector,
                    "method": obs.method,
                }
            )

        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Observation error: {str(e)}")
        return f"Failed to observe '{query}': {str(e)}"


def browser_screenshot(filename: str = "screenshot.png") -> str:
    """
    Take a screenshot of the current web page.

    Args:
        filename (str, optional): The filename to save the screenshot to.
                                 Defaults to "screenshot.png". 
                                 .png extension will be added automatically if not provided.

    Returns:
        str: Success message with the filename where screenshot was saved,
             or error message if screenshot fails

    Raises:
        RuntimeError: If browser is not initialized or page is not available
        Exception: If screenshot capture or file saving fails

    Example:
        >>> result = browser_screenshot()
        >>> print(result)
        "Screenshot saved to screenshot.png"

        >>> result = browser_screenshot("page_capture.png")
        >>> print(result)
        "Screenshot saved to page_capture.png"
    """
    return asyncio.run(_browser_screenshot_async(filename))


async def _browser_screenshot_async(filename: str) -> str:
    """Async implementation of browser_screenshot."""
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


def close_browser() -> str:
    """
    Close the browser when done with automation tasks.

    Returns:
        str: Success message if browser is closed successfully,
             or error message if closing fails

    Raises:
        Exception: If browser closing process encounters errors

    Example:
        >>> result = close_browser()
        >>> print(result)
        "Browser closed successfully"
    """
    return asyncio.run(_close_browser_async())


async def _close_browser_async() -> str:
    """Async implementation of close_browser."""
    try:
        await browser_state.close()
        return "Browser closed successfully"
    except Exception as e:
        logger.error(f"Close browser error: {str(e)}")
        return f"Failed to close browser: {str(e)}"


# Example usage
if __name__ == "__main__":
    # Create a Swarms agent with browser tools
    browser_agent = Agent(
        agent_name="BrowserAutomationAgent",
        model_name="gpt-4o-mini",
        max_loops=1,
        tools=[
            navigate_browser,
            browser_act,
            browser_extract,
            browser_observe,
            browser_screenshot,
            close_browser,
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
