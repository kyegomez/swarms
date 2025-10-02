# Mergers & Acquisition (M&A) Advisory Swarm

The M&A Advisory Swarm is a sophisticated multi-agent system designed to automate and streamline the entire mergers & acquisitions advisory workflow. By orchestrating a series of specialized AI agents, this swarm provides comprehensive analysis from initial intake to final recommendation.

## What it Does
The `MAAdvisorySwarm` operates as a **sequential workflow**, where each agent's output builds upon previous analyses, ensuring a cohesive and comprehensive advisory process. The swarm consists of the following agents:

| Agent Name | Agent (Name) | Key Responsibilities |
|-----------|--------------|---------------------|
| Intake & Scoping | Emma | Gathers essential information about the potential deal, including deal type, industry, target profile, objectives, timeline, budget, and specific concerns. It generates an initial Deal Brief. |
| Market & Strategic Analysis | Marcus | Evaluates industry dynamics, competitive landscape, and strategic fit. It leverages the `exa_search` tool to gather real-time market intelligence on trends, key players, and external factors. |
| Financial Valuation & Risk Assessment | Sophia | Performs comprehensive financial health analysis, various valuation methodologies (comparable companies, precedent transactions, DCF), synergy assessment, and a detailed risk assessment (financial, operational, legal, market). |
| Deal Structuring | David | Recommends the optimal transaction structure, considering asset vs. stock purchase, cash vs. stock consideration, earnouts, financing strategies, tax optimization, and deal protection mechanisms. |
| Integration Planning | Nathan | Develops a comprehensive post-merger integration roadmap, including Day 1 priorities, a 100-day plan, functional integration strategies (operations, systems, sales, HR), and synergy realization timelines. |
| Final Recommendation | Alex | Synthesizes all prior agent analyses into a comprehensive, executive-ready M&A Advisory Report, including an executive summary, investment thesis, key risks, deal structure, integration approach, and a clear GO/NO-GO/CONDITIONAL recommendation. |

## How to Set Up

To set up and run the M&A Advisory Swarm, follow these steps:

### Prerequisites

*   Python 3.8+
*   An Exa API Key (for the `exa_search` tool)

### Installation

1.  **Install dependencies:**
    The `ma_advisory.py` script relies on several libraries. These can be installed using the `requirements.txt` file located at the root of your project:
    ```bash
    pip install -r requirements.txt
    ```
    This will install `httpx`, `python-dotenv`, `loguru`, and other necessary packages.

2.  **Set up Exa API Key:**
    The `Market & Strategic Analysis Agent` utilizes the `exa_search` tool, which requires an `EXA_API_KEY`.
    Create a `.env` file in the root directory of your project (or wherever your application loads environment variables) and add your Exa API key:
    ```
    EXA_API_KEY="YOUR_EXA_API_KEY"
    ```
    Replace `"YOUR_EXA_API_KEY"` with your actual Exa API key.

## How to Run

Navigate to the `examples/demos/apps/` directory and run the `ma_advisory.py` script.

```bash
python examples/demos/apps/ma_advisory.py
```

OR, you can execute the following Python code directly (ensure all dependencies and the `.env` file are correctly set up):

```python
from typing import List
import os
from dotenv import load_dotenv
from loguru import logger
import httpx
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import history_output_formatter
from swarms.utils.any_to_str import any_to_str

# --- Exa Search Tool Integration ---
def exa_search(
    query: str,
    characters: int = 1000,  # Increased for more detailed M&A research
    sources: int = 5,  # More sources for comprehensive analysis
) -> str:
    """
    Perform a highly summarized Exa web search for M&A market intelligence.

    Args:
        query (str): Search query for M&A research.
        characters (int): Max characters for summary.
        sources (int): Number of sources.

    Returns:
        str: Condensed summary of search results.
    """
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        raise ValueError("EXA_API_KEY environment variable is not set")

    headers = {
        "x-api-key": api_key,
        "content-type": "application/json",
    }

    payload = {
        "query": query,
        "type": "auto",
        "numResults": sources,
        "contents": {
            "text": True,
            "summary": {
                "schema": {
                    "type": "object",
                    "required": ["answer"],
                    "additionalProperties": False,
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "Highly condensed summary of the M&A research result",
                        }
                    },
                }
            },
            "context": {"maxCharacters": characters},
        },
    }

    try:
        logger.info(f"[SEARCH] Exa M&A research: {query[:50]}...")
        response = httpx.post(
            "https://api.exa.ai/search",
            json=payload,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        json_data = response.json()
        return any_to_str(json_data)
    except Exception as e:
        logger.error(f"Exa search failed: {e}")
        return f"Search failed: {str(e)}. Please try again."

# Load environment variables
load_dotenv()

# System prompts for each agent
INTAKE_AGENT_PROMPT = """
You are an M&A Intake Specialist responsible for gathering comprehensive information about a potential transaction.

ROLE:
Engage with the user to understand the full context of the potential M&A deal, extracting critical details that will guide subsequent analyses.

RESPONSIBILITIES:
- Conduct a thorough initial interview to understand:
  * Transaction type (acquisition, merger, divestiture)
  * Industry and sector specifics
  * Target company profile and size
  * Strategic objectives
  * Buyer/seller perspective
  * Timeline and urgency
  * Budget constraints
  * Specific concerns or focus areas

OUTPUT FORMAT:
Provide a comprehensive Deal Brief that includes:
1. Transaction Overview
   - Proposed transaction type
   - Key parties involved
   - Initial strategic rationale

2. Stakeholder Context
   - Buyer's background and motivations
   - Target company's current position
   - Key decision-makers

3. Initial Assessment
   - Preliminary strategic fit
   - Potential challenges or red flags
   - Recommended focus areas for deeper analysis

4. Information Gaps
   - Questions that need further clarification
   - Additional data points required

IMPORTANT:
- Be thorough and systematic
- Ask probing questions to uncover nuanced details
- Maintain a neutral, professional tone
- Prepare a foundation for subsequent in-depth analysis
"""

MARKET_ANALYSIS_PROMPT = """
You are an M&A Market Intelligence Analyst tasked with conducting comprehensive market research.

ROLE:
Perform an in-depth analysis of market dynamics, competitive landscape, and strategic implications for the potential transaction.

TOOLS:
You have access to the exa_search tool for gathering real-time market intelligence.

RESPONSIBILITIES:
1. Conduct Market Research
   - Use exa_search to gather current market insights
   - Analyze industry trends, size, and growth potential
   - Identify key players and market share distribution

2. Competitive Landscape Analysis
   - Map out competitive ecosystem
   - Assess target company's market positioning
   - Identify potential competitive advantages or vulnerabilities

3. Strategic Fit Evaluation
   - Analyze alignment with buyer's strategic objectives
   - Assess potential market entry or expansion opportunities
   - Evaluate potential for market disruption

4. External Factor Assessment
   - Examine regulatory environment
   - Analyze technological disruption potential
   - Consider macroeconomic impacts

OUTPUT FORMAT:
Provide a comprehensive Market Analysis Report:
1. Market Overview
   - Market size and growth trajectory
   - Key industry trends
   - Competitive landscape summary

2. Strategic Fit Assessment
   - Market attractiveness score (1-10)
   - Strategic alignment evaluation
   - Potential synergies and opportunities

3. Risk and Opportunity Mapping
   - Key market opportunities
   - Potential competitive threats
   - Regulatory and technological risk factors

4. Recommended Next Steps
   - Areas requiring deeper investigation
   - Initial strategic recommendations
"""

FINANCIAL_VALUATION_PROMPT = """
You are an M&A Financial Analysis and Risk Expert. Perform comprehensive financial evaluation and risk assessment.

RESPONSIBILITIES:
1. Financial Health Analysis
   - Analyze revenue trends and quality
   - Evaluate profitability metrics (EBITDA, margins)
   - Conduct cash flow analysis
   - Assess balance sheet strength
   - Review working capital requirements

2. Valuation Analysis
   - Perform comparable company analysis
   - Conduct precedent transaction analysis
   - Develop Discounted Cash Flow (DCF) model
   - Assess asset-based valuation

3. Synergy and Risk Assessment
   - Quantify potential revenue and cost synergies
   - Identify financial and operational risks
   - Evaluate integration complexity
   - Assess potential deal-breakers

OUTPUT FORMAT:
1. Comprehensive Financial Analysis Report
2. Valuation Range (low, mid, high scenarios)
3. Synergy Potential Breakdown
4. Detailed Risk Matrix
5. Recommended Pricing Strategy
"""

DEAL_STRUCTURING_PROMPT = """
You are an M&A Deal Structuring Advisor. Recommend the optimal transaction structure.

RESPONSIBILITIES:
1. Transaction Structure Design
   - Evaluate asset vs stock purchase options
   - Analyze cash vs stock consideration
   - Design earnout provisions
   - Develop contingent payment structures

2. Financing Strategy
   - Recommend debt/equity mix
   - Identify optimal financing sources
   - Assess impact on buyer's capital structure

3. Tax and Legal Optimization
   - Design tax-efficient structure
   - Consider jurisdictional implications
   - Minimize tax liabilities

4. Deal Protection Mechanisms
   - Develop escrow arrangements
   - Design representations and warranties
   - Create indemnification provisions
   - Recommend non-compete agreements

OUTPUT FORMAT:
1. Recommended Deal Structure
2. Detailed Payment Terms
3. Key Contractual Protections
4. Tax Optimization Strategy
5. Rationale for Proposed Structure
"""

INTEGRATION_PLANNING_PROMPT = """
You are an M&A Integration Planning Expert. Develop a comprehensive post-merger integration roadmap.

RESPONSIBILITIES:
1. Immediate Integration Priorities
   - Define critical day-1 actions
   - Develop communication strategy
   - Identify quick win opportunities

2. 100-Day Integration Plan
   - Design organizational structure alignment
   - Establish governance framework
   - Create detailed integration milestones

3. Functional Integration Strategy
   - Plan operations consolidation
   - Design systems and technology integration
   - Align sales and marketing approaches
   - Develop cultural integration plan

4. Synergy Realization
   - Create detailed synergy capture timeline
   - Establish performance tracking mechanisms
   - Define accountability framework

OUTPUT FORMAT:
1. Comprehensive Integration Roadmap
2. Detailed 100-Day Plan
3. Functional Integration Strategies
4. Synergy Realization Timeline
5. Risk Mitigation Recommendations
"""

FINAL_RECOMMENDATION_PROMPT = """
You are the Senior M&A Advisory Partner. Synthesize all analyses into a comprehensive recommendation.

RESPONSIBILITIES:
1. Executive Summary
   - Summarize transaction overview
   - Highlight strategic rationale
   - Articulate key value drivers

2. Investment Thesis Validation
   - Assess strategic benefits
   - Evaluate financial attractiveness
   - Project long-term potential

3. Comprehensive Risk Assessment
   - Summarize top risks
   - Provide mitigation strategies
   - Identify potential deal-breakers

4. Final Recommendation
   - Provide clear GO/NO-GO recommendation
   - Specify recommended offer range
   - Outline key proceeding conditions

OUTPUT FORMAT:
1. Executive-Level Recommendation Report
2. Decision Framework
3. Risk-Adjusted Strategic Perspective
4. Actionable Next Steps
5. Recommendation Confidence Level
"""

class MAAdvisorySwarm:
    def __init__(
        self,
        name: str = "M&A Advisory Swarm",
        description: str = "Comprehensive AI-driven M&A advisory system",
        max_loops: int = 1,
        user_name: str = "M&A Advisor",
        output_type: str = "json",
    ):
        self.max_loops = max_loops
        self.name = name
        self.description = description
        self.user_name = user_name
        self.output_type = output_type
        
        self.agents = self._initialize_agents()
        self.conversation = Conversation()
        self.exa_search_results = []
        self.search_queries = []
        self.current_iteration = 0
        self.max_iterations = 1  # Limiting to 1 iteration for full sequential demo
        self.analysis_concluded = False
        
        self.handle_initial_processing()

    def handle_initial_processing(self):
        self.conversation.add(
            role="System",
            content=f"Company: {self.name}\n"
                    f"Description: {self.description}\n"
                    f"Mission: Provide comprehensive M&A advisory for {self.user_name}"
        )

    def _initialize_agents(self) -> List[Agent]:
        return [
            Agent(
                agent_name="Emma-Intake-Specialist",
                agent_description="Gathers comprehensive initial information about the potential M&A transaction.",
                system_prompt=INTAKE_AGENT_PROMPT,
                max_loops=self.max_loops,
                dynamic_temperature_enabled=True,
                output_type="final",
            ),
            Agent(
                agent_name="Marcus-Market-Analyst",
                agent_description="Conducts in-depth market research and competitive analysis.",
                system_prompt=MARKET_ANALYSIS_PROMPT,
                max_loops=self.max_loops,
                dynamic_temperature_enabled=True,
                output_type="final",
            ),
            Agent(
                agent_name="Sophia-Financial-Analyst",
                agent_description="Performs comprehensive financial valuation and risk assessment.",
                system_prompt=FINANCIAL_VALUATION_PROMPT,
                max_loops=self.max_loops,
                dynamic_temperature_enabled=True,
                output_type="final",
            ),
            Agent(
                agent_name="David-Deal-Structuring-Advisor",
                agent_description="Recommends optimal deal structure and terms.",
                system_prompt=DEAL_STRUCTURING_PROMPT,
                max_loops=self.max_loops,
                dynamic_temperature_enabled=True,
                output_type="final",
            ),
            Agent(
                agent_name="Nathan-Integration-Planner",
                agent_description="Develops comprehensive post-merger integration roadmap.",
                system_prompt=INTEGRATION_PLANNING_PROMPT,
                max_loops=self.max_loops,
                dynamic_temperature_enabled=True,
                output_type="final",
            ),
            Agent(
                agent_name="Alex-Final-Recommendation-Partner",
                agent_description="Synthesizes all analyses into a comprehensive recommendation.",
                system_prompt=FINAL_RECOMMENDATION_PROMPT,
                max_loops=self.max_loops,
                dynamic_temperature_enabled=True,
                output_type="final",
            )
        ]

    def find_agent_by_name(self, name: str) -> Agent:
        for agent in self.agents:
            if name in agent.agent_name:
                return agent
        return None

    def intake_and_scoping(self, user_input: str):
        """Phase 1: Intake and initial deal scoping"""
        emma_agent = self.find_agent_by_name("Intake-Specialist")
        
        emma_output = emma_agent.run(
            f"User Input: {user_input}\n\n"
            f"Conversation History: {self.conversation.get_str()}\n\n"
            f"Analyze the potential M&A transaction, extract key details, and prepare a comprehensive deal brief. "
            f"If information is unclear, ask clarifying questions."
        )
        
        self.conversation.add(
            role="Intake-Specialist", content=emma_output
        )
        
        # Extract potential search queries for market research
        self.search_queries = self._extract_search_queries(emma_output)
        
        return emma_output

    def _extract_search_queries(self, intake_output: str) -> List[str]:
        """Extract search queries from Intake Specialist output"""
        queries = []
        lines = intake_output.split('\n')
        
        # Look for lines that could be good search queries
        for line in lines:
            line = line.strip()
            # Simple heuristic: lines with potential research keywords
            if any(keyword in line.lower() for keyword in ['market', 'industry', 'trend', 'competitor', 'analysis']):
                if len(line) > 20:  # Ensure query is substantial
                    queries.append(line)
        
        # Fallback queries if none found
        if not queries:
            queries = [
                "M&A trends in technology sector",
                "Market analysis for potential business acquisition",
                "Competitive landscape in enterprise software"
            ]
        
        return queries[:3]  # Limit to 3 queries

    def market_research(self):
        """Phase 2: Conduct market research using exa_search"""
        # Execute exa_search for each query
        self.exa_search_results = []
        for query in self.search_queries:
            result = exa_search(query)
            self.exa_search_results.append({
                "query": query,
                "exa_result": result
            })
        
        # Pass results to Market Analysis agent
        marcus_agent = self.find_agent_by_name("Market-Analyst")
        
        # Build exa context
        exa_context = "\n\n[Exa Market Research Results]\n"
        for item in self.exa_search_results:
            exa_context += f"Query: {item['query']}\nResults: {item['exa_result']}\n\n"
        
        marcus_output = marcus_agent.run(
            f"Conversation History: {self.conversation.get_str()}\n\n"
            f"{exa_context}\n"
            f"Analyze these market research results. Provide comprehensive market intelligence and strategic insights."
        )
        
        self.conversation.add(
            role="Market-Analyst", content=marcus_output
        )
        
        return marcus_output

    def financial_valuation(self):
        """Phase 3: Perform comprehensive financial valuation and risk assessment"""
        sophia_agent = self.find_agent_by_name("Financial-Analyst")
        
        sophia_output = sophia_agent.run(
            f"Conversation History: {self.conversation.get_str()}\n\n"
            f"Perform comprehensive financial analysis and risk assessment based on previous insights."
        )
        
        self.conversation.add(
            role="Financial-Analyst", content=sophia_output
        )
        
        return sophia_output

    def deal_structuring(self):
        """Phase 4: Recommend optimal deal structure"""
        david_agent = self.find_agent_by_name("Deal-Structuring-Advisor")
        
        david_output = david_agent.run(
            f"Conversation History: {self.conversation.get_str()}\n\n"
            f"Recommend the optimal transaction structure and terms based on all prior analyses."
        )
        
        self.conversation.add(
            role="Deal-Structuring-Advisor", content=david_output
        )
        
        return david_output

    def integration_planning(self):
        """Phase 5: Develop post-merger integration roadmap"""
        nathan_agent = self.find_agent_by_name("Integration-Planner")
        
        nathan_output = nathan_agent.run(
            f"Conversation History: {self.conversation.get_str()}\n\n"
            f"Create a comprehensive integration plan to realize deal value."
        )
        
        self.conversation.add(
            role="Integration-Planner", content=nathan_output
        )
        
        return nathan_output

    def final_recommendation(self):
        """Phase 6: Synthesize all analyses into a comprehensive recommendation"""
        alex_agent = self.find_agent_by_name("Final-Recommendation-Partner")
        
        alex_output = alex_agent.run(
            f"Conversation History: {self.conversation.get_str()}\n\n"
            f"Synthesize all agent analyses into a comprehensive, actionable M&A recommendation."
        )
        
        self.conversation.add(
            role="Final-Recommendation-Partner", content=alex_output
        )
        
        return alex_output


    def run(self, initial_user_input: str):
        """
        Run the M&A advisory swarm with continuous analysis.
        
        Args:
            initial_user_input: User's initial M&A transaction details
        """
        self.conversation.add(role=self.user_name, content=initial_user_input)
        
        while not self.analysis_concluded and self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            logger.info(f"Starting analysis iteration {self.current_iteration}")
            
            # Phase 1: Intake and Scoping
            print(f"\n{'='*60}")
            print("ITERATION - INTAKE AND SCOPING")
            print(f"{'='*60}\n")
            self.intake_and_scoping(initial_user_input)
            
            # Phase 2: Market Research (with exa_search)
            print(f"\n{'='*60}")
            print("ITERATION - MARKET RESEARCH")
            print(f"{'='*60}\n")
            self.market_research()
            
            # Phase 3: Financial Valuation
            print(f"\n{'='*60}")
            print("ITERATION - FINANCIAL VALUATION")
            print(f"{'='*60}\n")
            self.financial_valuation()
            
            # Phase 4: Deal Structuring
            print(f"\n{'='*60}")
            print("ITERATION - DEAL STRUCTURING")
            print(f"{'='*60}\n")
            self.deal_structuring()
            
            # Phase 5: Integration Planning
            print(f"\n{'='*60}")
            print("ITERATION - INTEGRATION PLANNING")
            print(f"{'='*60}\n")
            self.integration_planning()
            
            # Phase 6: Final Recommendation
            print(f"\n{'='*60}")
            print("ITERATION - FINAL RECOMMENDATION")
            print(f"{'='*60}\n")
            self.final_recommendation()
            
            # Conclude analysis after one full sequence for demo purposes
            self.analysis_concluded = True
        
        # Return formatted conversation history
        return history_output_formatter(
            self.conversation, type=self.output_type
        )

def main():
    """Main entry point for M&A advisory swarm"""
    
    # Example M&A transaction details
    transaction_details = """
    We are exploring a potential acquisition of DataPulse Analytics by TechNova Solutions.
    
    Transaction Context:
    - Buyer: TechNova Solutions (NASDAQ: TNVA) - $500M annual revenue enterprise software company
    - Target: DataPulse Analytics - Series B AI-driven analytics startup based in San Francisco
    - Primary Objectives:
      * Expand predictive analytics capabilities in healthcare and financial services
      * Accelerate AI-powered business intelligence product roadmap
      * Acquire top-tier machine learning engineering talent
    
    Key Considerations:
    - Deep integration of DataPulse's proprietary AI models into TechNova's existing platform
    - Retention of key DataPulse leadership and engineering team
    - Projected 3-year ROI and synergy potential
    - Regulatory and compliance alignment
    - Technology stack compatibility
    """
    
    # Initialize the swarm
    ma_advisory_swarm = MAAdvisorySwarm(
        name="AI-Powered M&A Advisory System",
        description="Comprehensive AI-driven M&A advisory and market intelligence platform",
        user_name="Corporate Development Team",
        output_type="json",
        max_loops=1,
    )
    
    # Run the swarm
    print("\n" + "="*60)
    print("INITIALIZING M&A ADVISORY SWARM")
    print("="*60 + "\n")
    
    ma_advisory_swarm.run(initial_user_input=transaction_details)

if __name__ == "__main__":
    main()
```

## How it Can Be Used for M&A

The M&A Advisory Swarm can be utilized for a variety of M&A tasks, providing an automated and efficient approach to complex deal workflows:

*   **Automated Deal Scoping**: Quickly gather and structure initial information about a potential transaction.
*   **Real-time Market Intelligence**: Leverage web search capabilities to rapidly research industry trends, competitive landscapes, and strategic fit.
*   **Comprehensive Financial & Risk Analysis**: Perform detailed financial evaluations, valuation modeling, synergy assessments, and identify critical risks.
*   **Optimized Deal Structuring**: Recommend the most advantageous transaction structures, financing strategies, and deal protection mechanisms.
*   **Proactive Integration Planning**: Develop robust integration roadmaps to ensure seamless post-merger transitions and value realization.
*   **Executive-Ready Recommendations**: Synthesize complex analyses into clear, actionable recommendations for decision-makers.

By chaining these specialized agents, the M&A Advisory Swarm provides an end-to-end solution for corporate development teams, investment bankers, and M&A professionals, reducing manual effort and increasing the speed and quality of strategic decision-making.

## Contributing to Swarms

| Platform | Link | Description |
| :--------- | :----- | :------------ |
| 📚 Documentation | [docs.swarms.world](https://docs.swarms.world) | Official documentation and guides |
| 📝 Blog | [Medium](https://medium.com/@kyeg) | Latest updates and technical articles |
| 💬 Discord | [Join Discord](https://discord.gg/EamjgSaEQf) | Live chat and community support |
| 🐦 Twitter | [@kyegomez](https://twitter.com/kyegomez) | Latest news and announcements |
| 👥 LinkedIn | [The Swarm Corporation](https://www.linkedin.com/company/the-swarm-corporation) | Professional network and updates |
| 📺 YouTube | [Swarms Channel](https://www.youtube.com/channel/UC9yXyitkbU_WSy7bd_41SqQ) | Tutorials and demos |
| 🎫 Events | [Sign up here](https://lu.ma/5p2jnc2v) | Join our community events |