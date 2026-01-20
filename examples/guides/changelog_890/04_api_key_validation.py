"""
API Key Validation System Example

This example demonstrates the new check_swarms_api_key function that validates
API credentials before agents attempt to make requests. This proactive validation
prevents runtime errors and provides clear, actionable feedback when configuration issues exist.

Key features:
- Proactive validation before API requests
- Clear error messages for missing or invalid keys
- Guidance for obtaining and setting API keys
- Prevents runtime failures due to authentication issues
"""

from swarms.utils.swarms_marketplace_utils import check_swarms_api_key
from swarms import Agent

# Example 1: Demonstrate successful API key validation
try:
    api_key = check_swarms_api_key()

    # Create an agent that uses marketplace prompts
    agent = Agent(
        agent_name="MarketplaceAgent",
        marketplace_prompt_id="550e8400-e29b-41d4-a716-446655440000",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    print(
        "API key validation successful - agent created with marketplace integration"
    )

except ValueError as e:
    print(f"API key validation failed: {e}")
