#!/usr/bin/env python3
"""
Reproduce the exact code from GitHub issue #936 to test the fix.
"""

from typing import List
import http.client
import json
from swarms import Agent
import os


def get_realtor_data_from_one_source(location: str):
    """
    Fetch rental property data from the Realtor API for a specified location.

    Args:
        location (str): The location to search for rental properties (e.g., "Menlo Park, CA")

    Returns:
        str: JSON-formatted string containing rental property data

    Raises:
        http.client.HTTPException: If the API request fails
        json.JSONDecodeError: If the response cannot be parsed as JSON
    """
    # Mock implementation since we don't have API key
    return json.dumps({
        "properties": [
            {
                "name": f"Sample Property in {location}",
                "address": f"123 Main St, {location}",
                "price": 2800,
                "bedrooms": 2,
                "bathrooms": 1
            }
        ]
    }, indent=2)


def get_realtor_data_from_multiple_sources(locations: List[str]):
    """
    Fetch rental property data from multiple sources for a specified location.

    Args:
        location (List[str]): List of locations to search for rental properties (e.g., ["Menlo Park, CA", "Palo Alto, CA"])
    """
    output = []
    for location in locations:
        data = get_realtor_data_from_one_source(location)
        output.append(data)
    return output


def test_original_issue():
    """Test the exact scenario from the GitHub issue"""
    print("🧪 Testing Original Issue #936 Code...")
    
    agent = Agent(
        agent_name="Rental-Property-Specialist",
        system_prompt="""
        You are an expert rental property specialist with deep expertise in real estate analysis and tenant matching. Your core responsibilities include:
    1. Property Analysis & Evaluation
       - Analyze rental property features and amenities
       - Evaluate location benefits and drawbacks
       - Assess property condition and maintenance needs
       - Compare rental rates with market standards
       - Review lease terms and conditions
       - Identify potential red flags or issues

    2. Location Assessment
       - Analyze neighborhood safety and demographics
       - Evaluate proximity to amenities (schools, shopping, transit)
       - Research local market trends and development plans
       - Consider noise levels and traffic patterns
       - Assess parking availability and restrictions
       - Review zoning regulations and restrictions

    3. Financial Analysis
       - Calculate price-to-rent ratios
       - Analyze utility costs and included services
       - Evaluate security deposit requirements
       - Consider additional fees (pet rent, parking, etc.)
       - Compare with similar properties in the area
       - Assess potential for rent increases

    4. Tenant Matching
       - Match properties to tenant requirements
       - Consider commute distances
       - Evaluate pet policies and restrictions
       - Assess lease term flexibility
       - Review application requirements
       - Consider special accommodations needed

    5. Documentation & Compliance
       - Review lease agreement terms
       - Verify property certifications
       - Check compliance with local regulations
       - Assess insurance requirements
       - Review maintenance responsibilities
       - Document property condition

    When analyzing properties, always consider:
    - Value for money
    - Location quality
    - Property condition
    - Lease terms fairness
    - Safety and security
    - Maintenance and management quality
    - Future market potential
    - Tenant satisfaction factors

    Provide clear, objective analysis while maintaining professional standards and ethical considerations.""",
        model_name="claude-3-sonnet-20240229",
        max_loops=2,
        verbose=True,
        streaming_on=True,  # THIS WAS CAUSING THE ISSUE
        print_on=True,
        tools=[get_realtor_data_from_multiple_sources],
        api_key=os.getenv("ANTHROPIC_API_KEY"),  # Use appropriate API key
    )

    task = "What are the best properties in Menlo park and palo alto for rent under 3,000$"
    
    try:
        print(f"📝 Running task: {task}")
        print("🔄 With streaming=True and tools enabled...")
        
        result = agent.run(task)
        
        print(f"\n✅ Result: {result}")
        
        # Check if tool was executed
        memory_history = agent.short_memory.return_history_as_string()
        
        if "Tool Executor" in memory_history:
            print("\n✅ SUCCESS: Tool was executed successfully with streaming enabled!")
            print("🎉 Issue #936 appears to be FIXED!")
            return True
        else:
            print("\n❌ FAILURE: Tool execution was not detected")
            print("🚨 Issue #936 is NOT fixed yet")
            print("\nMemory History:")
            print(memory_history)
            return False
            
    except Exception as e:
        print(f"\n❌ FAILURE: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🔧 Testing the Exact Code from GitHub Issue #936")
    print("=" * 60)
    
    # Check if API key is available
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY are set.")
        print("Setting a dummy key for testing purposes...")
        os.environ["ANTHROPIC_API_KEY"] = "dummy-key-for-testing"
    
    success = test_original_issue()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 SUCCESS: The original issue appears to be RESOLVED!")
        print("✅ Agent tool usage now works with streaming enabled")
        print("✅ Tool execution logging is now present")
    else:
        print("❌ FAILURE: The original issue is NOT fully resolved yet")
        print("🔧 Additional fixes may be needed")