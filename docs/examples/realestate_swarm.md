# Real Estate Swarm

The Real Estate Swarm is a multi-agent system designed to automate and streamline the entire real estate transaction workflow. From lead generation to property maintenance, this swarm orchestrates a series of specialized AI agents to handle various aspects of buying, selling, and managing properties.

## What it Does

The `RealEstateSwarm` operates as a **sequential workflow**, where each agent's output feeds into the next, ensuring a cohesive and comprehensive process. The swarm consists of the following agents:

1.  **Lead Generation Agent (Alex)**: Identifies and qualifies potential real estate clients by gathering their property requirements, budget, preferred locations, and investment goals. This agent also uses the `exa_search` tool for initial lead information.
2.  **Property Research Agent (Emma)**: Conducts in-depth research on properties matching client criteria and market trends. It leverages the `exa_search` tool to gather up-to-date information on local market trends, property values, investment potential, and neighborhood insights.
3.  **Marketing Agent (Jack)**: Develops and executes marketing strategies to promote properties. This includes creating compelling listings, implementing digital marketing campaigns, and managing client interactions.
4.  **Transaction Management Agent (Sophia)**: Handles all documentation, legal, and financial aspects of property transactions, ensuring compliance and smooth closing processes.
5.  **Property Maintenance Agent (Michael)**: Manages property condition, oversees maintenance and repairs, and prepares properties for sale or rental, including staging and enhancing curb appeal.

## How to Set Up

To set up and run the Real Estate Swarm, follow these steps:

## Step 1: Setup and Installation

### Prerequisites

| Requirement           |
|-----------------------|
| Python 3.8 or higher  |
| pip package manager   |

1.  **Install dependencies:**
    Use the following command to download all dependencies.
    ```bash
    # Install Swarms framework
    pip install swarms

    # Install environment and logging dependencies
    pip install python-dotenv loguru

    # Install HTTP client and tools
    pip install httpx swarms_tools
    ```
2.  **Set up API Keys:**
    The `Property Research Agent` utilizes the `exa_search` tool, which requires an `EXA_API_KEY`.
    Create a `.env` file in the root directory of your project (or wherever your application loads environment variables) and add your API keys:
    ```
    EXA_API_KEY="YOUR_EXA_API_KEY"
    OPENAI_API_KEY="OPENAI_API_KEY"
    ```
    Replace `"YOUR_EXA_API_KEY"` & `"OPENAI_API_KEY"` with your actual API keys.

## Step 2: Running the Real Estate Swarm

```python

from swarms import Agent, SequentialWorkflow
import http.client
import json

def get_properties(postal_code: str, min_price: int, max_price: int, limit: int = 1) -> str:
    """
    Fetches real estate properties from Realty-in-US API using given zipcode, min price, and max price.
    All other payload fields remain constant.

    Returns the property's data as a string (JSON-encoded).
    """

    payload_dict = {
        "limit": limit,
        "offset": 0,
        "postal_code": postal_code,
        "status": ["for_sale", "ready_to_build"],
        "sort": {"direction": "desc", "field": "list_date"},
        "price_min": min_price,
        "price_max": max_price
    }

    payload = json.dumps(payload_dict)

    conn = http.client.HTTPSConnection("realty-in-us.p.rapidapi.com")
    headers = {
        "x-rapidapi-key": "35ae958601msh5c0eae51c54f989p1463c4jsn098ec5be18b8",
        "x-rapidapi-host": "realty-in-us.p.rapidapi.com",
        "Content-Type": "application/json"
    }

    conn.request("POST", "/properties/v3/list", payload, headers)
    res = conn.getresponse()
    data = res.read()
    decoded = data.decode("utf-8")
    try:
        result_dict = json.loads(decoded)
    except Exception:
        return decoded
    props_data = (
        result_dict.get("data", {})
        .get("home_search", {})
        .get("results", [])
    )
    if not props_data:
        return json.dumps({"error": "No properties found for that query."})
    return json.dumps(props_data[:limit])


REQUIREMENTS_ANALYZER_PROMPT = """
You are the Requirements Analyzer Agent for Real Estate.
ROLE:
Extract and clarify requirements from user input to create optimized property search queries.
RESPONSIBILITIES:
- Engage with the user to understand:
  * Desired property types and features
  * Required amenities and specifications
  * Preferred locations (city/area/zip)
  * Price/budget range
  * Timeline and purchase situation
  * Additional constraints or priorities
- Analyze user responses to identify:
  * Key search terms and must-have features
  * Priority factors in selection
  * Deal-breakers or constraints
  * Missing or unclear information to be clarified
- Generate search strategies:
  * Formulate 3-5 targeted search queries based on user requirements

OUTPUT FORMAT:
Provide a comprehensive requirements analysis:
1. User Profile Summary:
   - Property types/requirements of interest
   - Key features and specifications
   - Location and budget preferences
   - Priority factors
2. Search Strategy:
   - 3-5 optimized search queries (plain language, suitable for next agent's use)
   - Rationale for each query
   - Expected property/result types
3. Clarifications Needed:
   - Questions to refine search
   - Any missing info
IMPORTANT:
- INCLUDE all user responses verbatim in your analysis.
- Format queries clearly for the next agent.
- Ask follow-up questions if requirements are unclear.
"""

PROPERTY_RESEARCH_PROMPT = """
You are the Property Research Agent for Real Estate.
ROLE:
Conduct in-depth research on properties that match client criteria and market trends.

TOOLS:
You have access to get_properties. Use get_properties to find up-to-date and relevant information about properties for sale. Use ALL search queries produced by the previous agent (REQUIREMENTS_ANALYZER) as arguments to get_properties.
RESPONSIBILITIES:
- Perform property research using get_properties:
  * Seek properties by each proposed query and shortlist promising results.
  * Analyze each result by price, location, features, and comparables.
  * Highlight market trends if apparent from results.
  * Assess investment or suitability potential.
- Structure and cite property search findings.

OUTPUT FORMAT:
Provide a structured property research report:
1. Shortlist of matching properties (show summaries of each from get_properties results)
2. Detailed property analysis for each option
3. Insights on price, area, trends
4. Investment or suitability assessment
5. Recommendations for client
INSTRUCTIONS:
- Always use get_properties for up-to-date listing info; do not fabricate.
- Clearly indicate which properties are found from which query.
"""

MARKETING_PROMPT = """
You are the Marketing Agent for Real Estate.
ROLE:
Develop and execute marketing strategies to promote properties and attract potential buyers.
RESPONSIBILITIES:
- Create compelling property listings:
  * Professional photography
  * Detailed property descriptions
  * Highlight unique selling points
- Implement digital marketing strategies:
  * Social media campaigns
  * Email marketing
  * Online property platforms
  * Targeted advertising
- Manage client interactions:
  * Respond to property inquiries
  * Schedule property viewings
  * Facilitate initial negotiations
OUTPUT FORMAT:
Provide a comprehensive marketing report:
1. Marketing strategy overview
2. Property listing details
3. Marketing channel performance
4. Client inquiry and viewing logs
5. Initial negotiation summaries
"""

TRANSACTION_MANAGEMENT_PROMPT = """
You are the Transaction Management Agent for Real Estate.
ROLE:
Handle all documentation, legal, and financial aspects of property transactions.
RESPONSIBILITIES:
- Manage transaction documentation:
  * Prepare purchase agreements
  * Coordinate legal paperwork
  * Ensure compliance with real estate regulations
- Facilitate transaction process:
  * Coordinate with attorneys
  * Liaise with lenders
  * Manage escrow processes
  * Coordinate property inspections
- Ensure smooth closing:
  * Verify all financial requirements
  * Coordinate final document signings
  * Manage fund transfers
OUTPUT FORMAT:
Provide a detailed transaction management report:
1. Transaction document status
2. Legal and financial coordination details
3. Inspection and verification logs
4. Closing process timeline
5. Recommendations for transaction completion
"""

PROPERTY_MAINTENANCE_PROMPT = """
You are the Property Maintenance Agent for Real Estate.
ROLE:
Manage property condition, maintenance, and preparation for sale or rental.
RESPONSIBILITIES:
- Conduct regular property inspections:
  * Assess property condition
  * Identify maintenance needs
  * Ensure safety standards
- Coordinate maintenance and repairs:
  * Hire and manage contractors
  * Oversee repair and renovation work
  * Manage landscaping and cleaning
- Prepare properties for market:
  * Stage properties
  * Enhance curb appeal
  * Recommend cost-effective improvements
OUTPUT FORMAT:
Provide a comprehensive property maintenance report:
1. Inspection findings
2. Maintenance and repair logs
3. Improvement recommendations
4. Property readiness status
5. Contractor and service provider details
"""

def main():
    user_requirements = """
    I'm looking for a spacious 3-bedroom apartment with modern amenities.
    - Price: $1,000,000 - $1,500,000
    - Location: Downtown Manhattan, Upper East Side
    - Investment: Long-term, high ROI preferred
    - Contact: john.doe@email.com, +1-555-123-4567
    - Timeline: Within the next 3 months
    - Financials: Pre-approved for mortgage
    """

    agents = [
        Agent(
            agent_name="Alex-Requirements-Analyzer",
            agent_description="Analyzes user property requirements and creates optimized property search queries.",
            system_prompt=REQUIREMENTS_ANALYZER_PROMPT,
            model_name="gpt-4.1",
            max_loops=1,
            temperature=0.7,
        ),
        Agent(
            agent_name="Emma-Property-Research",
            agent_description="Conducts comprehensive property search and market analysis.",
            system_prompt=PROPERTY_RESEARCH_PROMPT,
            model_name="gpt-4.1",
            max_loops=1,
            temperature=0.7,
            tools=[get_properties],
        ),
        Agent(
            agent_name="Jack-Marketing",
            agent_description="Develops and executes marketing strategies for properties.",
            system_prompt=MARKETING_PROMPT,
            model_name="gpt-4.1",
            max_loops=1,
            temperature=0.7,
        ),
        Agent(
            agent_name="Sophia-Transaction-Management",
            agent_description="Handles legal, financial, and document aspects of property transactions.",
            system_prompt=TRANSACTION_MANAGEMENT_PROMPT,
            model_name="gpt-4.1",
            max_loops=1,
            temperature=0.7,
        ),
        Agent(
            agent_name="Michael-Property-Maintenance",
            agent_description="Oversees property condition, maintenance, and market readiness.",
            system_prompt=PROPERTY_MAINTENANCE_PROMPT,
            model_name="gpt-4.1",
            max_loops=1,
            temperature=0.7,
        ),
    ]

    workflow = SequentialWorkflow(
        name="real-estate-sequential-workflow",
        agents=agents,
        max_loops=1,
        team_awareness=True,
    )

    workflow.run(user_requirements)

if __name__ == "__main__":
    main()

```

## How it Can Be Used for Real Estate

The Real Estate Swarm can be utilized for a variety of real estate tasks, providing an automated and efficient approach to complex workflows:

*   **Automated Lead Qualification**: Automatically gather and assess potential client needs and financial readiness.
*   **Comprehensive Property Analysis**: Rapidly research and generate detailed reports on properties and market trends using real-time web search capabilities.
*   **Streamlined Marketing**: Develop and execute marketing strategies, including listing creation and social media campaigns.
*   **Efficient Transaction Management**: Automate the handling of legal documents, financial coordination, and closing processes.
*   **Proactive Property Maintenance**: Manage property upkeep and prepare assets for optimal market presentation.

By chaining these specialized agents, the Real Estate Swarm provides an end-to-end solution for real estate professionals, reducing manual effort and increasing operational efficiency.

## Contributing to Swarms
| Platform | Link | Description |
|----------|------|-------------|
| 📚 Documentation | [docs.swarms.world](https://docs.swarms.world) | Official documentation and guides |
| 📝 Blog | [Medium](https://medium.com/@kyeg) | Latest updates and technical articles |
| 💬 Discord | [Join Discord](https://discord.gg/EamjgSaEQf) | Live chat and community support |
| 🐦 Twitter | [@kyegomez](https://twitter.com/kyegomez) | Latest news and announcements |
| 👥 LinkedIn | [The Swarm Corporation](https://www.linkedin.com/company/the-swarm-corporation) | Professional network and updates |
| 📺 YouTube | [Swarms Channel](https://www.youtube.com/channel/UC9yXyitkbU_WSy7bd_41SqQ) | Tutorials and demos |
| 🎫 Events | [Sign up here](https://lu.ma/5p2jnc2v) | Join our community events |
