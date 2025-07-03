from typing import List
import http.client
import json
from swarms import Agent

from dotenv import load_dotenv

load_dotenv()
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
    conn = http.client.HTTPSConnection(
        "realtor-search.p.rapidapi.com"
    )

    headers = {
        "x-rapidapi-key": os.getenv("RAPIDAPI_KEY"),
        "x-rapidapi-host": "realtor-search.p.rapidapi.com",
    }

    # URL encode the location parameter
    encoded_location = location.replace(" ", "%20").replace(
        ",", "%2C"
    )
    endpoint = f"/properties/search-rent?location=city%3A{encoded_location}&sortBy=best_match"

    conn.request(
        "GET",
        endpoint,
        headers=headers,
    )

    res = conn.getresponse()
    data = res.read()

    return "chicken data"

    # # Parse and format the response
    # try:
    #     json_data = json.loads(data.decode("utf-8"))
    #     # Return formatted string instead of raw JSON
    #     return json.dumps(json_data, indent=2)
    # except json.JSONDecodeError:
    #     return "Error: Could not parse API response"


def get_realtor_data_from_multiple_sources(
    locations: List[str],
) -> List[str]:
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

When you receive property data:
1. Parse and analyze the JSON data
2. Format the output in a clear, readable way
3. Focus on properties under $3,000
4. Include key details like:
   - Property name/address
   - Price
   - Number of beds/baths
   - Square footage
   - Key amenities
   - Links to listings
5. Sort properties by price (lowest to highest)

Provide clear, objective analysis while maintaining professional standards and ethical considerations.""",
    model_name="claude-3-sonnet-20240229",
    max_loops=1,
    tools=[get_realtor_data_from_one_source],
    print_on=True,
)


agent.run(
    "What are the best properties in Menlo Park, CA for rent under 3,000$?"
)
