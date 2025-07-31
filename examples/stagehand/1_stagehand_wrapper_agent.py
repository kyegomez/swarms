"""
Stagehand Browser Automation Agent for Swarms
=============================================

This example demonstrates how to create a Swarms-compatible agent
that wraps Stagehand's browser automation capabilities.

The StagehandAgent class inherits from the Swarms Agent base class
and implements browser automation through natural language commands.
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field

from swarms import Agent as SwarmsAgent
from stagehand import Stagehand, StagehandConfig

load_dotenv()


class WebData(BaseModel):
    """Schema for extracted web data."""

    url: str = Field(..., description="The URL of the page")
    title: str = Field(..., description="Page title")
    content: str = Field(..., description="Extracted content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class StagehandAgent(SwarmsAgent):
    """
    A Swarms agent that integrates Stagehand for browser automation.

    This agent can navigate websites, extract data, perform actions,
    and observe page elements using natural language instructions.
    """

    def __init__(
        self,
        agent_name: str = "StagehandBrowserAgent",
        browserbase_api_key: Optional[str] = None,
        browserbase_project_id: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        model_api_key: Optional[str] = None,
        env: str = "LOCAL",  # LOCAL or BROWSERBASE
        *args,
        **kwargs,
    ):
        """
        Initialize the StagehandAgent.

        Args:
            agent_name: Name of the agent
            browserbase_api_key: API key for Browserbase (if using cloud)
            browserbase_project_id: Project ID for Browserbase
            model_name: LLM model to use
            model_api_key: API key for the model
            env: Environment - LOCAL or BROWSERBASE
        """
        # Don't pass stagehand-specific args to parent
        super().__init__(agent_name=agent_name, *args, **kwargs)

        self.stagehand_config = StagehandConfig(
            env=env,
            api_key=browserbase_api_key
            or os.getenv("BROWSERBASE_API_KEY"),
            project_id=browserbase_project_id
            or os.getenv("BROWSERBASE_PROJECT_ID"),
            model_name=model_name,
            model_api_key=model_api_key or os.getenv("OPENAI_API_KEY"),
        )
        self.stagehand = None
        self._initialized = False

    async def _init_stagehand(self):
        """Initialize Stagehand instance."""
        if not self._initialized:
            self.stagehand = Stagehand(self.stagehand_config)
            await self.stagehand.init()
            self._initialized = True
            logger.info(f"Stagehand initialized for {self.agent_name}")

    async def _close_stagehand(self):
        """Close Stagehand instance."""
        if self.stagehand and self._initialized:
            await self.stagehand.close()
            self._initialized = False
            logger.info(f"Stagehand closed for {self.agent_name}")

    def run(self, task: str, *args, **kwargs) -> str:
        """
        Execute a browser automation task.

        The task string should contain instructions like:
        - "Navigate to example.com and extract the main content"
        - "Go to google.com and search for 'AI agents'"
        - "Extract all company names from https://ycombinator.com"

        Args:
            task: Natural language description of the browser task

        Returns:
            String result of the task execution
        """
        return asyncio.run(self._async_run(task, *args, **kwargs))

    async def _async_run(
        self, task: str, *args, **kwargs
    ) -> str:
        """Async implementation of run method."""
        try:
            await self._init_stagehand()

            # Parse the task to determine actions
            result = await self._execute_browser_task(task)

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error in browser task: {str(e)}")
            return f"Error executing browser task: {str(e)}"
        finally:
            # Keep browser open for potential follow-up tasks
            pass

    async def _execute_browser_task(
        self, task: str
    ) -> Dict[str, Any]:
        """
        Execute a browser task based on natural language instructions.

        This method interprets the task and calls appropriate Stagehand methods.
        """
        page = self.stagehand.page
        result = {"task": task, "status": "completed", "data": {}}

        # Determine if task involves navigation
        if any(
            keyword in task.lower()
            for keyword in ["navigate", "go to", "visit", "open"]
        ):
            # Extract URL from task
            import re

            url_pattern = r"https?://[^\s]+"
            urls = re.findall(url_pattern, task)
            if not urls and any(
                domain in task for domain in [".com", ".org", ".net"]
            ):
                # Try to extract domain names
                domain_pattern = r"(\w+\.\w+)"
                domains = re.findall(domain_pattern, task)
                if domains:
                    urls = [f"https://{domain}" for domain in domains]

            if urls:
                url = urls[0]
                await page.goto(url)
                result["data"]["navigated_to"] = url
                logger.info(f"Navigated to {url}")

        # Determine action type
        if "extract" in task.lower():
            # Perform extraction
            extraction_prompt = task.replace("extract", "").strip()
            extracted = await page.extract(extraction_prompt)
            result["data"]["extracted"] = extracted
            result["action"] = "extract"

        elif "click" in task.lower() or "press" in task.lower():
            # Perform action
            action_result = await page.act(task)
            result["data"]["action_performed"] = str(action_result)
            result["action"] = "act"

        elif "search" in task.lower():
            # Perform search action
            search_query = task.split("search for")[-1].strip().strip("'\"")
            # First, find the search box
            search_box = await page.observe("find the search input field")
            if search_box:
                # Click on search box and type
                await page.act(f"click on {search_box[0]}")
                await page.act(f"type '{search_query}'")
                await page.act("press Enter")
                result["data"]["search_query"] = search_query
                result["action"] = "search"

        elif "observe" in task.lower() or "find" in task.lower():
            # Perform observation
            observation = await page.observe(task)
            result["data"]["observation"] = [
                {"description": obs.description, "selector": obs.selector}
                for obs in observation
            ]
            result["action"] = "observe"

        else:
            # General action
            action_result = await page.act(task)
            result["data"]["action_result"] = str(action_result)
            result["action"] = "general"

        return result

    def cleanup(self):
        """Clean up browser resources."""
        if self._initialized:
            asyncio.run(self._close_stagehand())

    def __del__(self):
        """Ensure browser is closed on deletion."""
        self.cleanup()


# Example usage
if __name__ == "__main__":
    # Create a Stagehand browser agent
    browser_agent = StagehandAgent(
        agent_name="WebScraperAgent",
        model_name="gpt-4o-mini",
        env="LOCAL",  # Use LOCAL for Playwright, BROWSERBASE for cloud
    )

    # Example 1: Navigate and extract data
    print("Example 1: Basic navigation and extraction")
    result1 = browser_agent.run(
        "Navigate to https://news.ycombinator.com and extract the titles of the top 5 stories"
    )
    print(result1)
    print("\n" + "=" * 50 + "\n")

    # Example 2: Perform a search
    print("Example 2: Search on a website")
    result2 = browser_agent.run(
        "Go to google.com and search for 'Swarms AI framework'"
    )
    print(result2)
    print("\n" + "=" * 50 + "\n")

    # Example 3: Extract structured data
    print("Example 3: Extract specific information")
    result3 = browser_agent.run(
        "Navigate to https://example.com and extract the main heading and first paragraph"
    )
    print(result3)

    # Clean up
    browser_agent.cleanup()