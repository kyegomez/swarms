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
from swarms import Agent, SequentialWorkflow
import http.client
import json
import urllib.parse

def get_jobs(query: str, limit: int = 10) -> str:
    """
    Fetches real-time jobs using JSearch API based on role, location, and experience.
    Uses http.client to match verified working example.
    """
    # Prepare query string for URL
    encoded_query = urllib.parse.quote(query)
    path = f"/search?query={encoded_query}&page=1&num_pages=1&country=us&limit={limit}&date_posted=all"
    
    conn = http.client.HTTPSConnection("jsearch.p.rapidapi.com")

    headers = {
        "x-rapidapi-key": "", #<------- Add your RapidAPI key here otherwise it will not work
        "x-rapidapi-host": "jsearch.p.rapidapi.com"
    }

    conn.request("GET", path, headers=headers)

    res = conn.getresponse()
    data = res.read()
    decoded = data.decode("utf-8")
    try:
        result_dict = json.loads(decoded)
    except Exception:
        # fallback for unexpected output
        return decoded

    results = result_dict.get("data", [])
    jobs_list = [
        {
            "title": job.get("job_title"),
            "company": job.get("employer_name"),
            "location": job.get("job_city") or job.get("job_country"),
            "experience": job.get("job_required_experience", {}).get("required_experience_in_months"),
            "url": job.get("job_apply_link")
        }
        for job in results
    ]
    return json.dumps(jobs_list)


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
  
OUTPUT FORMAT:
Provide a comprehensive requirements analysis:
1. User Profile Summary:
   - Job titles of interest
   - Key skills and qualifications
   - Location preferences
   - Salary range
   - Priority factors

2. Search Strategy:
   - List of 3-5 optimized search queries, formatted EXACTLY for linkedin.com/jobs/search/?keywords=...
   - Rationale for each query
   - Expected result types

3. Clarifications Needed (if any):
   - Questions to refine search
   - Missing information

IMPORTANT:
- Always include ALL user responses verbatim in your analysis
- Format search queries clearly for the next agent and fit directly to LinkedIn search URLs
- Be specific and actionable in your recommendations
- Ask follow-up questions if requirements are unclear
"""

SEARCH_EXECUTOR_PROMPT = """
You are the Search Executor Agent for Job Search.

ROLE:
Your job is to execute a job search by querying the tool EXACTLY ONCE using the following required format (FILL IN WHERE IT HAS [ ] WITH THE QUERY INFO OTHERWISE STATED):

The argument for the query is to be provided as a plain text string in the following format (DO NOT include technical addresses, just the core query string):

    [jobrole] jobs in [geographiclocation/remoteorinpersonorhybrid] 

For example:
    developer jobs in chicago
    senior product manager jobs in remote
    data engineer jobs in new york hybrid

TOOLS:
You have access to three tools:
- get_jobs: helps find open job opportunities for your specific job and requirements.

RESPONSIBILITIES:
- Run ONE single query, in the above format, as the argument to get_jobs.
- Analyze search results for:
  * Job title match
  * Skills alignment
  * Location compatibility
  * Salary range fit
  * Company reputation
  * Role responsibilities
  * Growth opportunities

- Categorize each result into one of:
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
1. Search Execution Summary:
   - The query executed (write ONLY the string argument supplied, e.g., "developer jobs in chicago" or "software engineer jobs in new york remote")
   - Total results found
   - Distribution by match category

2. Detailed Job Listings (grouped by match strength):
   For each job:
   - Company and Job Title
   - Location and Work Type
   - Key Requirements
   - Why it's a match (or not)
   - Match Score (percentage)
   - Application link
   - Source (specify get_jobs)

3. Search Insights:
   - Common trends/themes in the results
   - Gaps between results and requirements
   - Market observations

INSTRUCTIONS:
- Run only the single query in the format described above, with no extra path, no technical addresses, and no full URLs.
- Use all three tools, as applicable, with that exact query argument.
- Clearly cite which results come from which source.
- Be objective in match assessment.
- Provide actionable, structured insights.
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

def main():
    # User input for job requirements
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

    # Define your agents in a list as in the example format
    agents = [
        Agent(
            agent_name="Sarah-Requirements-Analyzer",
            agent_description="Analyzes user requirements and creates optimized job search queries.",
            system_prompt=REQUIREMENTS_ANALYZER_PROMPT,
            model_name="gpt-4.1",
            max_loops=1,
            temperature=0.7,
        ),
        Agent(
            agent_name="David-Search-Executor",
            agent_description="Executes job searches and analyzes results for relevance.",
            system_prompt=SEARCH_EXECUTOR_PROMPT,
            model_name="gpt-4.1",
            max_loops=1,
            temperature=0.7,
            tools=[get_jobs],
        ),
        Agent(
            agent_name="Lisa-Results-Curator",
            agent_description="Curates and presents job results for user decision-making.",
            system_prompt=RESULTS_CURATOR_PROMPT,
            model_name="gpt-4.1",
            max_loops=1,
            temperature=0.7,
        ),
    ]

    # Setup the SequentialWorkflow pipeline (following the style of the ETF example)
    workflow = SequentialWorkflow(
        name="job-search-sequential-workflow",
        agents=agents,
        max_loops=1,
        team_awareness=True,
    )

    workflow.run(user_requirements)

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
