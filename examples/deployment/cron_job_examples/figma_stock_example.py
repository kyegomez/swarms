"""
Example script demonstrating how to fetch Figma (FIG) stock data using Yahoo Finance.
"""

from cron_job_examples.cron_job_example import (
    get_figma_stock_data,
    get_figma_stock_data_simple,
)
from loguru import logger
import json


def main():
    """
    Main function to demonstrate Figma stock data fetching.
    """
    logger.info("Starting Figma stock data demonstration")

    try:
        # Example 1: Get comprehensive data as dictionary
        logger.info("Fetching comprehensive Figma stock data...")
        figma_data = get_figma_stock_data()

        # Print the data in a structured format
        print("\n" + "=" * 50)
        print("COMPREHENSIVE FIGMA STOCK DATA")
        print("=" * 50)
        print(json.dumps(figma_data, indent=2, default=str))

        # Example 2: Get simple formatted data
        logger.info("Fetching simple formatted Figma stock data...")
        simple_data = get_figma_stock_data_simple()

        print("\n" + "=" * 50)
        print("SIMPLE FORMATTED FIGMA STOCK DATA")
        print("=" * 50)
        print(simple_data)

        # Example 3: Access specific data points
        logger.info("Accessing specific data points...")

        current_price = figma_data["current_market_data"][
            "current_price"
        ]
        market_cap = figma_data["current_market_data"]["market_cap"]
        pe_ratio = figma_data["financial_metrics"]["pe_ratio"]

        print("\nKey Metrics:")
        print(f"Current Price: ${current_price}")
        print(f"Market Cap: ${market_cap:,}")
        print(f"P/E Ratio: {pe_ratio}")

        # Example 4: Check if stock is performing well
        price_change = figma_data["current_market_data"][
            "price_change"
        ]
        if isinstance(price_change, (int, float)):
            if price_change > 0:
                print(
                    f"\n📈 Figma stock is up ${price_change:.2f} today!"
                )
            elif price_change < 0:
                print(
                    f"\n📉 Figma stock is down ${abs(price_change):.2f} today."
                )
            else:
                print("\n➡️ Figma stock is unchanged today.")

        logger.info(
            "Figma stock data demonstration completed successfully!"
        )

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
