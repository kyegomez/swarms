"""
Example script demonstrating how to fetch Figma (FIG) stock data using swarms_tools Yahoo Finance API.
This shows the alternative approach using the existing swarms_tools package.
"""

from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)
from swarms_tools import yahoo_finance_api
from loguru import logger
import json


def get_figma_data_with_swarms_tools():
    """
    Fetches Figma stock data using the swarms_tools Yahoo Finance API.

    Returns:
        dict: Figma stock data from swarms_tools
    """
    try:
        logger.info("Fetching Figma stock data using swarms_tools...")
        figma_data = yahoo_finance_api(["FIG"])
        return figma_data
    except Exception as e:
        logger.error(f"Error fetching data with swarms_tools: {e}")
        raise


def analyze_figma_with_agent():
    """
    Uses a Swarms agent to analyze Figma stock data.
    """
    try:
        # Initialize the agent with Yahoo Finance tool
        agent = Agent(
            agent_name="Figma-Analysis-Agent",
            agent_description="Specialized agent for analyzing Figma stock data",
            system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
            max_loops=1,
            model_name="gpt-4o-mini",
            tools=[yahoo_finance_api],
            dynamic_temperature_enabled=True,
        )

        # Ask the agent to analyze Figma
        analysis = agent.run(
            "Analyze the current stock data for Figma (FIG) and provide insights on its performance, valuation metrics, and recent trends."
        )

        return analysis

    except Exception as e:
        logger.error(f"Error in agent analysis: {e}")
        raise


def main():
    """
    Main function to demonstrate different approaches for Figma stock data.
    """
    logger.info("Starting Figma stock analysis with swarms_tools")

    try:
        # Method 1: Direct API call
        print("\n" + "=" * 60)
        print("METHOD 1: Direct swarms_tools API call")
        print("=" * 60)

        figma_data = get_figma_data_with_swarms_tools()
        print("Raw data from swarms_tools:")
        print(json.dumps(figma_data, indent=2, default=str))

        # Method 2: Agent-based analysis
        print("\n" + "=" * 60)
        print("METHOD 2: Agent-based analysis")
        print("=" * 60)

        analysis = analyze_figma_with_agent()
        print("Agent analysis:")
        print(analysis)

        # Method 3: Comparison with custom function
        print("\n" + "=" * 60)
        print("METHOD 3: Comparison with custom function")
        print("=" * 60)

        from cron_job_examples.cron_job_example import (
            get_figma_stock_data_simple,
        )

        custom_data = get_figma_stock_data_simple()
        print("Custom function output:")
        print(custom_data)

        logger.info("All methods completed successfully!")

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
