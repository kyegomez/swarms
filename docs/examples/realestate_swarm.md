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

### Prerequisites

*   Python 3.8+
*   Poetry (or pip) for dependency management
*   An Exa API Key (for the `exa_search` tool)

### Installation
1.  **Install dependencies:**
    Use the following command to download all dependencies.
    ```bash
    pip install -r requirements.txt
    ```
2.  **Set up Exa API Key:**
    The `Property Research Agent` utilizes the `exa_search` tool, which requires an `EXA_API_KEY`.
    Create a `.env` file in the root directory of your project (or wherever your application loads environment variables) and add your Exa API key:
    ```
    EXA_API_KEY="YOUR_EXA_API_KEY"
    ```
    Replace `"YOUR_EXA_API_KEY"` with your actual Exa API key.

## How to Run

Navigate to the `examples/demos/real_estate/` directory and run the `realestate_swarm.py` script.

OR run the following code

```python

from typing import List

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import history_output_formatter

# --- Exa Search Tool Integration ---
# Import and define exa_search as a callable tool for property research.
from swarms_tools import exa_search
# System prompts for each agent

LEAD_GENERATION_PROMPT = """
You are the Lead Generation Agent for Real Estate.

ROLE:
Collect potential leads for real estate transactions by identifying buyers, sellers, and investors through various channels.

RESPONSIBILITIES:
- Identify potential clients through:
  * Real estate websites
  * Social media platforms
  * Referral networks
  * Local community events
- Conduct initial consultations to understand:
  * Client's property requirements
  * Budget constraints
  * Preferred locations
  * Investment goals
- Qualify leads by assessing:
  * Financial readiness
  * Specific property needs
  * Urgency of transaction

OUTPUT FORMAT:
Provide a comprehensive lead report that includes:
1. Client profile and contact information
2. Detailed requirements and preferences
3. Initial assessment of client's real estate goals
4. Qualification status
5. Recommended next steps

IMPORTANT CONTEXT SHARING:
When preparing the lead report, clearly summarize and include all answers and information provided by the user. Integrate these user responses directly into your analysis and the lead report. This ensures that when your report is sent to the next agent, it contains all relevant user preferences, requirements, and context needed for further research and decision-making.

REMEMBER:
- Ensure the user's answers are explicitly included in your report so the next agent can use them for property research and analysis.
"""

PROPERTY_RESEARCH_PROMPT = """
You are the Property Research Agent for Real Estate.
ROLE:
Conduct in-depth research on properties that match client criteria and market trends.

TOOLS:
You have access to the exa_search tool. Use exa_search to find up-to-date and relevant information about properties, market trends, and neighborhood data. Leverage the answers provided by the user and the outputs from previous agents to formulate your search queries. Always use exa_search to supplement your research and validate your findings.

RESPONSIBILITIES:
- Perform comprehensive property market analysis using exa_search:
  * Local market trends
  * Property value assessments
  * Investment potential
  * Neighborhood evaluations
- Research properties matching client specifications (using both user answers and previous agent outputs):
  * Price range
  * Location preferences
  * Property type
  * Specific amenities
- Compile detailed property reports including:
  * Comparative market analysis (use exa_search for recent comps)
  * Property history
  * Potential appreciation
  * Neighborhood insights (gathered via exa_search)

INSTRUCTIONS:
- Always use exa_search to find the most current and relevant information for your analysis.
- Formulate your exa_search queries based on the user's answers and the outputs from previous agents.
- Clearly indicate in your report where exa_search was used to obtain information.

OUTPUT FORMAT:
Provide a structured property research report:
1. Shortlist of matching properties (include sources from exa_search)
2. Detailed property analysis for each option (cite exa_search findings)
3. Market trend insights (supported by exa_search data)
4. Investment potential assessment
5. Recommendations for client consideration

REMEMBER:
Do not rely solely on prior knowledge. Always use exa_search to verify and enhance your research with the latest available data.
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

class RealEstateSwarm:
    def __init__(
        self,
        name: str = "Real Estate Swarm",
        description: str = "A comprehensive AI-driven real estate transaction workflow",
        max_loops: int = 1,
        user_name: str = "Real Estate Manager",
        property_type: str = "Residential",
        output_type: str = "json",
        user_lead_info: str = "",
    ):
        self.max_loops = max_loops
        self.name = name
        self.description = description
        self.user_name = user_name
        self.property_type = property_type
        self.output_type = output_type
        self.user_lead_info = user_lead_info

        self.agents = self._initialize_agents()
        self.conversation = Conversation()
        self.handle_initial_processing()
        self.exa_search_results = []  # Store exa_search results for property research

    def handle_initial_processing(self):
        self.conversation.add(
            role=self.user_name,
            content=f"Company: {self.name}\n"
                    f"Description: {self.description}\n"
                    f"Property Type: {self.property_type}"
        )

    def _initialize_agents(self) -> List[Agent]:
        return [
            Agent(
                agent_name="Alex-Lead-Generation",
                agent_description="Identifies and qualifies potential real estate clients across various channels.",
                system_prompt=LEAD_GENERATION_PROMPT,
                max_loops=self.max_loops,
                dynamic_temperature_enabled=True,
                output_type="final",
            ),
            Agent(
                agent_name="Emma-Property-Research",
                agent_description="Conducts comprehensive property research and market analysis.",
                system_prompt=PROPERTY_RESEARCH_PROMPT,
                max_loops=self.max_loops,
                dynamic_temperature_enabled=True,
                output_type="final",
            ),
            Agent(
                agent_name="Jack-Marketing",
                agent_description="Develops and executes marketing strategies for properties.",
                system_prompt=MARKETING_PROMPT,
                max_loops=self.max_loops,
                dynamic_temperature_enabled=True,
                output_type="final",
            ),
            Agent(
                agent_name="Sophia-Transaction-Management",
                agent_description="Manages legal and financial aspects of real estate transactions.",
                system_prompt=TRANSACTION_MANAGEMENT_PROMPT,
                max_loops=self.max_loops,
                dynamic_temperature_enabled=True,
                output_type="final",
            ),
            Agent(
                agent_name="Michael-Property-Maintenance",
                agent_description="Oversees property condition, maintenance, and market preparation.",
                system_prompt=PROPERTY_MAINTENANCE_PROMPT,
                max_loops=self.max_loops,
                dynamic_temperature_enabled=True,
                output_type="final",
            ),
        ]

    def find_agent_by_name(self, name: str) -> Agent:
        for agent in self.agents:
            if name in agent.agent_name:
                return agent
        return None

    def lead_generation(self):
        alex_agent = self.find_agent_by_name("Lead-Generation")
        # Directly inject the user_lead_info into the prompt for the first agent
        alex_output = alex_agent.run(
            f"User Lead Information:\n{self.user_lead_info}\n\n"
            f"History: {self.conversation.get_str()}\n"
            f"Generate leads for {self.property_type} real estate transactions. Identify potential clients and their specific requirements."
        )
        self.conversation.add(
            role="Lead-Generation", content=alex_output
        )

        # --- After lead generation, use user_lead_info as queries for exa_search ---
        queries = []
        if isinstance(self.user_lead_info, list):
            queries = [str(q) for q in self.user_lead_info if str(q).strip()]
        elif isinstance(self.user_lead_info, str):
            if "\n" in self.user_lead_info:
                queries = [q.strip() for q in self.user_lead_info.split("\n") if q.strip()]
            else:
                queries = [self.user_lead_info.strip()] if self.user_lead_info.strip() else []

        self.exa_search_results = []
        for q in queries:
            result = exa_search(q)
            self.exa_search_results.append({
                "query": q,
                "exa_result": result
            })

    def property_research(self):
        emma_agent = self.find_agent_by_name("Property-Research")
        # Pass ALL exa_search results as direct context to the property research agent
        exa_context = ""
        if hasattr(self, "exa_search_results") and self.exa_search_results:
            # Directly inject all exa_search results as context
            exa_context = "\n\n[Exa Search Results]\n"
            for item in self.exa_search_results:
                exa_context += f"Query: {item['query']}\nExa Search Result: {item['exa_result']}\n"

        emma_output = emma_agent.run(
            f"History: {self.conversation.get_str()}\n"
            f"{exa_context}"
            f"Conduct research on {self.property_type} properties, analyze market trends, and prepare a comprehensive property report."
        )
        self.conversation.add(
            role="Property-Research", content=emma_output
        )

    def property_marketing(self):
        jack_agent = self.find_agent_by_name("Marketing")
        jack_output = jack_agent.run(
            f"History: {self.conversation.get_str()}\n"
            f"Develop marketing strategies for {self.property_type} properties, create listings, and manage client interactions."
        )
        self.conversation.add(
            role="Marketing", content=jack_output
        )

    def transaction_management(self):
        sophia_agent = self.find_agent_by_name("Transaction-Management")
        sophia_output = sophia_agent.run(
            f"History: {self.conversation.get_str()}\n"
            f"Manage legal and financial aspects of the {self.property_type} property transaction."
        )
        self.conversation.add(
            role="Transaction-Management", content=sophia_output
        )

    def property_maintenance(self):
        michael_agent = self.find_agent_by_name("Property-Maintenance")
        michael_output = michael_agent.run(
            f"History: {self.conversation.get_str()}\n"
            f"Assess and manage maintenance for the {self.property_type} property to prepare it for market."
        )
        self.conversation.add(
            role="Property-Maintenance", content=michael_output
        )

    def run(self, task: str):
        """
        Process the real estate workflow through the swarm, coordinating tasks among agents.
        """
        self.conversation.add(role=self.user_name, content=task)

        # Execute workflow stages
        self.lead_generation()
        self.property_research()
        self.property_marketing()
        self.transaction_management()
        self.property_maintenance()

        return history_output_formatter(
            self.conversation, type=self.output_type
        )

def main():
    """FOR ACTUAL INPUT FROM USER"""
    # # Collect user lead information at the beginning
    # print("Please provide the following information for lead generation:")
    # questions = [
    #     "What are the client's property requirements?",
    #     "What is the client's budget range or constraints?",
    #     "What are the client's preferred locations?",
    #     "What are the client's investment goals?",
    #     "What is the client's contact information?",
    #     "What is the urgency of the transaction?",
    #     "Is the client financially ready to proceed?"
    # ]

    # user_lead_info = ""
    # for q in questions:
    #     answer = input(f"{q}\n> ").strip()
    #     user_lead_info += f"Q: {q}\nA: {answer if answer else '[No answer provided]'}\n\n"
    # Pre-filled placeholder answers for each question
    user_lead_info = [
        "Spacious 3-bedroom apartment with modern amenities",
        "$1,000,000 - $1,500,000",
        "Downtown Manhattan, Upper East Side",
        "Long-term investment with high ROI",
        "john.doe@email.com, +1-555-123-4567",
        "Within the next 3 months",
        "Yes, pre-approved for mortgage"
    ]


    # Initialize the swarm
    real_estate_swarm = RealEstateSwarm(
        max_loops=1,
        name="Global Real Estate Solutions",
        description="Comprehensive AI-driven real estate transaction workflow",
        user_name="Real Estate Director",
        property_type="Luxury Residential",
        output_type="json",
        user_lead_info=user_lead_info,
    )

    # Sample real estate task
    sample_task = """
    We have a high-end luxury residential property in downtown Manhattan.
    Client requirements:
    - Sell the property quickly
    - Target high-net-worth individuals
    - Maximize property value
    - Ensure smooth and discreet transaction
    """
    # Run the swarm
    real_estate_swarm.run(task=sample_task)

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