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
    characters: int = 500,
    sources: int = 3,
) -> str:
    """
    Perform a highly summarized Exa web search for job listings and career information.

    Args:
        query (str): Search query for jobs or career info.
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
                            "description": "Highly condensed summary of the search result",
                        }
                    },
                }
            },
            "context": {"maxCharacters": characters},
        },
    }

    try:
        logger.info(f"[SEARCH] Exa job search: {query[:50]}...")
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

REQUIREMENTS_ANALYZER_PROMPT = """
You are the Requirements Analyzer Agent for Job Search.

ROLE:
Extract and clarify job search requirements from user input to create optimized search queries.

RESPONSIBILITIES:
- Engage with the user to understand:
  * Desired job titles and roles
  * Required skills and qualifications
  * Preferred locations (remote, hybrid, on-site)
  * Salary expectations
  * Company size and culture preferences
  * Industry preferences
  * Experience level
  * Work authorization status
  * Career goals and priorities

- Analyze user responses to identify:
  * Key search terms and keywords
  * Must-have vs nice-to-have requirements
  * Deal-breakers or constraints
  * Priority factors in job selection

- Generate optimized search queries:
  * Create 3-5 targeted search queries based on user requirements
  * Combine job titles, skills, locations, and key criteria
  * Format queries for maximum relevance

OUTPUT FORMAT:
Provide a comprehensive requirements analysis:
1. User Profile Summary:
   - Job titles of interest
   - Key skills and qualifications
   - Location preferences
   - Salary range
   - Priority factors

2. Search Strategy:
   - List of 3-5 optimized search queries
   - Rationale for each query
   - Expected result types

3. Clarifications Needed (if any):
   - Questions to refine search
   - Missing information

IMPORTANT:
- Always include ALL user responses verbatim in your analysis
- Format search queries clearly for the next agent
- Be specific and actionable in your recommendations
- Ask follow-up questions if requirements are unclear
"""

SEARCH_EXECUTOR_PROMPT = """
You are the Search Executor Agent for Job Search.

ROLE:
Execute job searches using exa_search and analyze results for relevance.

TOOLS:
You have access to the exa_search tool. Use it to find current job listings and career opportunities.

RESPONSIBILITIES:
- Execute searches using queries from the Requirements Analyzer
- Use exa_search for EACH query provided
- Analyze search results for:
  * Job title match
  * Skills alignment
  * Location compatibility
  * Salary range fit
  * Company reputation
  * Role responsibilities
  * Growth opportunities

- Categorize results:
  * Strong Match (80-100% alignment)
  * Good Match (60-79% alignment)
  * Moderate Match (40-59% alignment)
  * Weak Match (<40% alignment)

- For each job listing, extract:
  * Job title and company
  * Location and work arrangement
  * Key requirements
  * Salary range (if available)
  * Application link or contact
  * Match score and reasoning

OUTPUT FORMAT:
Provide structured search results:
1. Search Execution Summary:
   - Queries executed
   - Total results found
   - Distribution by match category

2. Detailed Job Listings (organized by match strength):
   For each job:
   - Company and Job Title
   - Location and Work Type
   - Key Requirements
   - Why it's a match (or not)
   - Match Score (percentage)
   - Application link
   - Source (cite exa_search)

3. Search Insights:
   - Common themes in results
   - Gap analysis (requirements not met)
   - Market observations

INSTRUCTIONS:
- Always use exa_search for EVERY query provided
- Cite exa_search results clearly
- Be objective in match assessment
- Provide actionable insights
"""

RESULTS_CURATOR_PROMPT = """
You are the Results Curator Agent for Job Search.

ROLE:
Filter, organize, and present job search results to the user for decision-making.

RESPONSIBILITIES:
- Review all search results from the Search Executor
- Filter and prioritize based on:
  * Match scores
  * User requirements
  * Application deadlines
  * Job quality indicators

- Organize results into:
  * Top Recommendations (top 3-5 best matches)
  * Strong Alternatives (next 5-10 options)
  * Worth Considering (other relevant matches)

- For top recommendations, provide:
  * Detailed comparison
  * Pros and cons for each
  * Application strategy suggestions
  * Next steps

- Engage user for feedback:
  * Present curated results clearly
  * Ask which jobs interest them
  * Identify what's missing
  * Determine if new search is needed

OUTPUT FORMAT:
Provide a curated job search report:

1. Executive Summary:
   - Total jobs reviewed
   - Number of strong matches
   - Key findings

2. Top Recommendations (detailed):
   For each (max 5):
   - Company & Title
   - Why it's a top match
   - Key highlights
   - Potential concerns
   - Recommendation strength (1-10)
   - Application priority (High/Medium/Low)

3. Strong Alternatives (brief list):
   - Company & Title
   - One-line match summary
   - Match score

4. User Decision Point:
   Ask the user:
   - "Which of these jobs interest you most?"
   - "What's missing from these results?"
   - "Should we refine the search or proceed with applications?"
   - "Any requirements you'd like to adjust?"

5. Next Steps:
   Based on user response, either:
   - Proceed with selected jobs
   - Run new search with adjusted criteria
   - Deep dive into specific opportunities

IMPORTANT:
- Make it easy for users to make decisions
- Be honest about job fit
- Provide clear paths forward
- Always ask for user feedback before concluding
"""

class JobSearchSwarm:
    def __init__(
        self,
        name: str = "AI Job Search Swarm",
        description: str = "An intelligent job search system that finds your ideal role",
        max_loops: int = 1,
        user_name: str = "Job Seeker",
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
        self.max_iterations = 10  # Prevent infinite loops
        self.search_concluded = False
        
        self.handle_initial_processing()

    def handle_initial_processing(self):
        self.conversation.add(
            role="System",
            content=f"Company: {self.name}\n"
                    f"Description: {self.description}\n"
                    f"Mission: Find the perfect job match for {self.user_name}"
        )

    def _initialize_agents(self) -> List[Agent]:
        return [
            Agent(
                agent_name="Sarah-Requirements-Analyzer",
                agent_description="Analyzes user requirements and creates optimized job search queries.",
                system_prompt=REQUIREMENTS_ANALYZER_PROMPT,
                max_loops=self.max_loops,
                dynamic_temperature_enabled=True,
                output_type="final",
            ),
            Agent(
                agent_name="David-Search-Executor",
                agent_description="Executes job searches and analyzes results for relevance.",
                system_prompt=SEARCH_EXECUTOR_PROMPT,
                max_loops=self.max_loops,
                dynamic_temperature_enabled=True,
                output_type="final",
            ),
            Agent(
                agent_name="Lisa-Results-Curator",
                agent_description="Curates and presents job results for user decision-making.",
                system_prompt=RESULTS_CURATOR_PROMPT,
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

    def analyze_requirements(self, user_input: str):
        """Phase 1: Analyze user requirements and generate search queries"""
        sarah_agent = self.find_agent_by_name("Requirements-Analyzer")
        
        sarah_output = sarah_agent.run(
            f"User Input: {user_input}\n\n"
            f"Conversation History: {self.conversation.get_str()}\n\n"
            f"Analyze the user's job search requirements and generate 3-5 optimized search queries. "
            f"If information is unclear, ask clarifying questions."
        )
        
        self.conversation.add(
            role="Requirements-Analyzer", content=sarah_output
        )
        
        # Extract search queries from Sarah's output
        self.search_queries = self._extract_search_queries(sarah_output)
        
        return sarah_output

    def _extract_search_queries(self, analyzer_output: str) -> List[str]:
        """Extract search queries from Requirements Analyzer output"""
        queries = []
        lines = analyzer_output.split('\n')
        
        # Look for lines that appear to be search queries
        for line in lines:
            line = line.strip()
            # Simple heuristic: lines with certain keywords or patterns
            if any(keyword in line.lower() for keyword in ['query:', 'search:', 'query']):
                # Extract the actual query
                if ':' in line:
                    query = line.split(':', 1)[1].strip()
                    if query and len(query) > 10:
                        queries.append(query)
        
        # If no queries found, create default ones based on common patterns
        if not queries:
            logger.warning("No explicit queries found, generating fallback queries")
            queries = [
                "software engineer jobs remote",
                "data scientist positions",
                "product manager opportunities"
            ]
        
        return queries[:5]  # Limit to 5 queries

    def execute_searches(self):
        """Phase 2: Execute searches using exa_search and analyze results"""
        # Execute exa_search for each query
        self.exa_search_results = []
        for query in self.search_queries:
            result = exa_search(query)
            self.exa_search_results.append({
                "query": query,
                "exa_result": result
            })
        
        # Pass results to Search Executor agent
        david_agent = self.find_agent_by_name("Search-Executor")
        
        # Build exa context
        exa_context = "\n\n[Exa Search Results]\n"
        for item in self.exa_search_results:
            exa_context += f"Query: {item['query']}\nResults: {item['exa_result']}\n\n"
        
        david_output = david_agent.run(
            f"Conversation History: {self.conversation.get_str()}\n\n"
            f"{exa_context}\n"
            f"Analyze these job search results. Categorize each job by match strength and provide detailed analysis."
        )
        
        self.conversation.add(
            role="Search-Executor", content=david_output
        )
        
        return david_output

    def curate_results(self) -> str:
        """Phase 3: Curate results and get user feedback"""
        lisa_agent = self.find_agent_by_name("Results-Curator")
        
        lisa_output = lisa_agent.run(
            f"Conversation History: {self.conversation.get_str()}\n\n"
            f"Curate the job search results, present top recommendations, and ask the user for feedback. "
            f"Determine if we should continue searching or if the user has found suitable options."
        )
        
        self.conversation.add(
            role="Results-Curator", content=lisa_output
        )
        
        return lisa_output

    def end(self) -> tuple[bool, str]:
        """
        Conclude the job search without user interaction.
        
        Returns:
            tuple[bool, str]: (needs_refinement, user_feedback)
        """
        return False, "Search completed successfully."

    def run(self, initial_user_input: str):
        """
        Run the job search swarm with continuous optimization.
        
        Args:
            initial_user_input: User's initial job search requirements
        """
        self.conversation.add(role=self.user_name, content=initial_user_input)
        
        user_input = initial_user_input
        
        while not self.search_concluded and self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            logger.info(f"Starting search iteration {self.current_iteration}")
            
            # Phase 1: Analyze requirements
            print(f"\n{'='*60}")
            print(f"ITERATION {self.current_iteration} - ANALYZING REQUIREMENTS")
            print(f"{'='*60}\n")
            self.analyze_requirements(user_input)

            
            # Phase 2: Execute searches
            print(f"\n{'='*60}")
            print(f"ITERATION {self.current_iteration} - EXECUTING JOB SEARCHES")
            print(f"{'='*60}\n")
            self.execute_searches()
            
            # Phase 3: Curate and present results
            print(f"\n{'='*60}")
            print(f"ITERATION {self.current_iteration} - CURATING RESULTS")
            print(f"{'='*60}\n")
            self.curate_results()
            
            # Phase 4: Get user feedback
            needs_refinement, user_feedback = self.end()
            
            # Add user feedback to conversation
            self.conversation.add(
                role=self.user_name, 
                content=f"User Feedback: {user_feedback}"
            )
            
            # Check if we should continue
            if not needs_refinement:
                self.search_concluded = True
                print(f"\n{'='*60}")
                print("SEARCH CONCLUDED - USER SATISFIED WITH RESULTS")
                print(f"{'='*60}\n")
            else:
                # In production, get new user input here
                print(f"\n{'='*60}")
                print("SEARCH REQUIRES REFINEMENT")
                print(f"{'='*60}\n")
                # For demo, we'll stop after first iteration
                self.search_concluded = True
        
        # Return formatted conversation history
        return history_output_formatter(
            self.conversation, type=self.output_type
        )

def main():
    """Main entry point for job search swarm"""
    
    # Example 1: Pre-filled user requirements (for testing)
    user_requirements = """
    I'm looking for a senior software engineer position with the following requirements:
    - Job Title: Senior Software Engineer or Staff Engineer
    - Skills: Python, distributed systems, cloud architecture (AWS/GCP), Kubernetes
    - Location: Remote (US-based) or San Francisco Bay Area
    - Salary: $180k - $250k
    - Company: Mid-size to large tech companies, prefer companies with strong engineering culture
    - Experience Level: 7+ years
    - Industry: SaaS, Cloud Infrastructure, or Developer Tools
    - Work Authorization: US Citizen
    - Priorities: Technical challenges, work-life balance, remote flexibility, equity upside
    - Deal-breakers: No pure management roles, no strict return-to-office policies
    """
    
    # Initialize the swarm
    job_search_swarm = JobSearchSwarm(
        name="AI-Powered Job Search Engine",
        description="Intelligent job search system that continuously refines results until the perfect match is found",
        user_name="Job Seeker",
        output_type="json",
        max_loops=1,
    )
    
    # Run the swarm
    print("\n" + "="*60)
    print("INITIALIZING JOB SEARCH SWARM")
    print("="*60 + "\n")
    
    job_search_swarm.run(initial_user_input=user_requirements)

if __name__ == "__main__":
    main()