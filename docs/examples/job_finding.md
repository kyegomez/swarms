# Job Finding Swarm

## Overview

The Job Finding Swarm is an intelligent multi-agent system designed to automate and streamline the job search process using the Swarms framework. It leverages specialized AI agents to analyze user requirements, execute comprehensive job searches, and curate relevant opportunities, transforming traditional job hunting into an intelligent, collaborative process.

## Key Components

The Job Finding Swarm consists of three specialized agents, each responsible for a critical stage of the job search process:

| Agent Name                  | Role                        | Responsibilities                                                                                  |
| :-------------------------- | :-------------------------- | :----------------------------------------------------------------------------------------------- |
| **Sarah-Requirements-Analyzer** | Clarifies requirements      | Gathers and analyzes user job preferences; generates 3-5 search queries.                         |
| **David-Search-Executor**   | Runs job searches           | Uses `exa_search` for each query; analyzes and categorizes job results by match strength.        |
| **Lisa-Results-Curator**    | Organizes results           | Filters, prioritizes, and presents jobs; provides top picks and refines search with user input.  |

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

## Step 2: Running the Job Finding Swarm

```python
from typing import List
from loguru import logger
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import history_output_formatter
from swarms_tools import exa_search

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

Upon execution, the swarm will:
1.  Analyze the provided `user_requirements`.
2.  Generate and execute search queries using Exa.
3.  Curate and present the results in a structured format, including top recommendations and a prompt for user feedback.

The output will be printed to the console, showing the progression of the agents through each phase of the job search.

## Workflow Stages

The `JobSearchSwarm` processes the job search through a continuous, iterative workflow:

1.  **Phase 1: Analyze Requirements (`analyze_requirements`)**: The `Sarah-Requirements-Analyzer` agent processes the user's input and conversation history to extract job criteria and generate optimized search queries.
2.  **Phase 2: Execute Searches (`execute_searches`)**: The `David-Search-Executor` agent takes the generated queries, uses the `exa_search` tool to find job listings, and analyzes their relevance against the user's requirements.
3.  **Phase 3: Curate Results (`curate_results`)**: The `Lisa-Results-Curator` agent reviews, filters, and organizes the search results, presenting top recommendations and asking for user feedback to guide further iterations.
4.  **Phase 4: Get User Feedback (`end`)**: In a full implementation, this stage gathers explicit user feedback to determine if the search needs refinement or can be concluded. For demonstration, this is a simulated step.

The swarm continues these phases in a loop until the `search_concluded` flag is set to `True` or `max_iterations` is reached.

## Customization

You can customize the `JobSearchSwarm` by modifying the `JobSearchSwarm` class parameters or the agents' prompts:

*   **`name` and `description`**: Customize the swarm's identity.
*   **`user_name`**: Define the name of the user interacting with the swarm.
*   **`output_type`**: Specify the desired output format for the conversation history (e.g., "json" or "list").
*   **`max_loops`**: Control the number of internal reasoning iterations each agent performs (set during agent initialization).
*   **`system_prompt`**: Modify the `REQUIREMENTS_ANALYZER_PROMPT`, `SEARCH_EXECUTOR_PROMPT`, and `RESULTS_CURATOR_PROMPT` to refine agent behavior and output.
*   **`max_iterations`**: Limit the total number of search cycles the swarm performs.

## Best Practices

To get the most out of the AI Job Finding Swarm:

*   **Provide Clear Requirements**: Start with a detailed and unambiguous `initial_user_input` to help the Requirements Analyzer generate effective queries.
*   **Iterate and Refine**: In a live application, leverage the user feedback loop to continuously refine search criteria and improve result relevance.
*   **Monitor Agent Outputs**: Regularly review the outputs from each agent to ensure they are performing as expected and to identify areas for prompt improvement.
*   **Manage API Usage**: Be mindful of your Exa API key usage, especially when experimenting with `max_iterations` or a large number of search queries.

## Limitations

*   **Prompt Engineering Dependency**: The quality of the search results heavily depends on the clarity and effectiveness of the agent `system_prompt`s and the initial user input.
*   **Exa Search Scope**: The `exa_search` tool's effectiveness is tied to the breadth and depth of Exa's indexed web content.
*   **Iteration Control**: The current `end` method in `examples/demos/apps/job_finding.py` is simplified for demonstration. A robust production system would require a more sophisticated user interaction mechanism to determine when to stop or refine the search.
*   **Verification Needed**: All AI-generated outputs, including job matches and summaries, should be independently verified by the user.
