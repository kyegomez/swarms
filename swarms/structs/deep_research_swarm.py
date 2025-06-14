import concurrent.futures
import json
import os
from typing import Any, List

from dotenv import load_dotenv
from rich.console import Console
import requests

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.formatter import formatter
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.str_to_dict import str_to_dict

console = Console()
load_dotenv()

# Number of worker threads for concurrent operations
MAX_WORKERS = (
    os.cpu_count() * 2
)  # Optimal number of workers based on CPU cores

###############################################################################
# 1. System Prompts for Each Scientist Agent
###############################################################################


def exa_search(query: str, **kwargs: Any) -> str:
    """Performs web search using Exa.ai API and returns formatted results."""
    api_url = "https://api.exa.ai/search"
    api_key = os.getenv("EXA_API_KEY")

    if not api_key:
        return "### Error\nEXA_API_KEY environment variable not set\n"

    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
    }

    safe_kwargs = {
        str(k): v
        for k, v in kwargs.items()
        if k is not None and v is not None and str(k) != "None"
    }

    payload = {
        "query": query,
        "useAutoprompt": True,
        "numResults": safe_kwargs.get("num_results", 10),
        "contents": {
            "text": True,
            "highlights": {"numSentences": 10},
        },
    }

    for key, value in safe_kwargs.items():
        if key not in payload and key not in [
            "query",
            "useAutoprompt",
            "numResults",
            "contents",
        ]:
            payload[key] = value

    try:
        response = requests.post(
            api_url, json=payload, headers=headers
        )
        if response.status_code != 200:
            return f"### Error\nHTTP {response.status_code}: {response.text}\n"
        json_data = response.json()
    except Exception as e:
        return f"### Error\n{str(e)}\n"

    if "error" in json_data:
        return f"### Error\n{json_data['error']}\n"

    formatted_text = []
    search_params = json_data.get("effectiveFilters", {})
    query = search_params.get("query", "General web search")
    formatted_text.append(
        f"### Exa Search Results for: '{query}'\n\n---\n"
    )

    results = json_data.get("results", [])
    if not results:
        formatted_text.append("No results found.\n")
        return "".join(formatted_text)

    for i, result in enumerate(results, 1):
        title = result.get("title", "No title")
        url = result.get("url", result.get("id", "No URL"))
        published_date = result.get("publishedDate", "")
        highlights = result.get("highlights", [])
        highlight_text = (
            "\n".join(
                (
                    h.get("text", str(h))
                    if isinstance(h, dict)
                    else str(h)
                )
                for h in highlights[:3]
            )
            if highlights
            else "No summary available"
        )

        formatted_text.extend(
            [
                f"{i}. **{title}**\n",
                f"   - URL: {url}\n",
                f"   - Published: {published_date.split('T')[0] if published_date else 'Date unknown'}\n",
                f"   - Key Points:\n      {highlight_text}\n\n",
            ]
        )

    return "".join(formatted_text)


# Define the research tools schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_topic",
            "description": "Conduct a thorough search on a specified topic or subtopic, generating a precise array of highly detailed search queries tailored to the input parameters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "depth": {
                        "type": "integer",
                        "description": "Indicates the level of thoroughness for the search. Values range from 1 to 3, where 1 signifies a superficial search and 3 indicates an in-depth exploration of the topic.",
                    },
                    "detailed_queries": {
                        "type": "array",
                        "description": "An array of specific search queries generated based on the input query and the specified depth. Each query must be crafted to elicit detailed and relevant information from various sources.",
                        "items": {
                            "type": "string",
                            "description": "Each item in this array must represent a unique search query targeting a specific aspect of the main topic, ensuring a comprehensive exploration of the subject matter.",
                        },
                    },
                },
                "required": ["depth", "detailed_queries"],
            },
        },
    },
]

RESEARCH_AGENT_PROMPT = """
You are an advanced research agent specialized in conducting deep, comprehensive research across multiple domains.
Your task is to:

1. Break down complex topics into searchable subtopics
2. Generate diverse search queries to explore each subtopic thoroughly
3. Identify connections and patterns across different areas of research
4. Synthesize findings into coherent insights
5. Identify gaps in current knowledge and suggest areas for further investigation

For each research task:
- Consider multiple perspectives and approaches
- Look for both supporting and contradicting evidence
- Evaluate the credibility and relevance of sources
- Track emerging trends and recent developments
- Consider cross-disciplinary implications

Output Format:
- Provide structured research plans
- Include specific search queries for each subtopic
- Prioritize queries based on relevance and potential impact
- Suggest follow-up areas for deeper investigation
"""

SUMMARIZATION_AGENT_PROMPT = """
You are an expert information synthesis and summarization agent designed for producing clear, accurate, and insightful summaries of complex information. Your core capabilities include:


Core Capabilities:
- Identify and extract key concepts, themes, and insights from any given content
- Recognize patterns, relationships, and hierarchies within information
- Filter out noise while preserving crucial context and nuance
- Handle multiple sources and perspectives simultaneously

Summarization Strategy
1. Multi-level Structure
   - Provide an extensive summary 
   - Follow with key findings
   - Include detailed insights with supporting evidence
   - End with implications or next steps when relevant

2. Quality Standards
   - Maintain factual accuracy and precision
   - Preserve important technical details and terminology
   - Avoid oversimplification of complex concepts
   - Include quantitative data when available
   - Cite or reference specific sources when summarizing claims

3. Clarity & Accessibility
   - Use clear, concise language
   - Define technical terms when necessary
   - Structure information logically
   - Use formatting to enhance readability
   - Maintain appropriate level of technical depth for the audience

4. Synthesis & Analysis
   - Identify conflicting information or viewpoints
   - Highlight consensus across sources
   - Note gaps or limitations in the information
   - Draw connections between related concepts
   - Provide context for better understanding

OUTPUT REQUIREMENTS:
- Begin with a clear statement of the topic or question being addressed
- Use consistent formatting and structure
- Clearly separate different levels of detail
- Include confidence levels for conclusions when appropriate
- Note any areas requiring additional research or clarification

Remember: Your goal is to make complex information accessible while maintaining accuracy and depth. Prioritize clarity without sacrificing important nuance or detail."""


class DeepResearchSwarm:
    def __init__(
        self,
        name: str = "DeepResearchSwarm",
        description: str = "A swarm that conducts comprehensive research across multiple domains",
        max_loops: int = 1,
        nice_print: bool = True,
        output_type: str = "json",
        max_workers: int = os.cpu_count()
        * 2,  # Let the system decide optimal thread count
        token_count: bool = False,
        research_model_name: str = "gpt-4o-mini",
        claude_summarization_model_name: str = "claude-3-5-sonnet-20240620",
    ):
        self.name = name
        self.description = description
        self.max_loops = max_loops
        self.nice_print = nice_print
        self.output_type = output_type
        self.max_workers = max_workers
        self.research_model_name = research_model_name
        self.claude_summarization_model_name = (
            claude_summarization_model_name
        )

        self.reliability_check()
        self.conversation = Conversation(token_count=token_count)

        # Create a persistent ThreadPoolExecutor for the lifetime of the swarm
        # This eliminates thread creation overhead on each query
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        )

        # Initialize the research agent
        self.research_agent = Agent(
            agent_name="Deep-Research-Agent",
            agent_description="Specialized agent for conducting comprehensive research across multiple domains",
            system_prompt=RESEARCH_AGENT_PROMPT,
            max_loops=1,  # Allow multiple iterations for thorough research
            tools_list_dictionary=tools,
            model_name=self.research_model_name,
            output_type="final",
        )

        self.summarization_agent = Agent(
            agent_name="Summarization-Agent",
            agent_description="Specialized agent for summarizing research results",
            system_prompt=SUMMARIZATION_AGENT_PROMPT,
            max_loops=1,
            model_name=self.claude_summarization_model_name,
            output_type="final",
        )

    def __del__(self):
        """Clean up the executor on object destruction"""
        self.executor.shutdown(wait=False)

    def reliability_check(self):
        """Check the reliability of the query"""
        if self.max_loops < 1:
            raise ValueError("max_loops must be greater than 0")

        formatter.print_panel(
            "DeepResearchSwarm is booting up...", "blue"
        )
        formatter.print_panel("Reliability check passed", "green")

    def get_queries(self, query: str) -> List[str]:
        """
        Generate a list of detailed search queries based on the input query.

        Args:
            query (str): The main research query to explore

        Returns:
            List[str]: A list of detailed search queries
        """
        self.conversation.add(role="User", content=query)

        # Get the agent's response
        agent_output = self.research_agent.run(query)

        # Transform the string into a list of dictionaries
        agent_output = json.loads(agent_output)
        print(agent_output)
        print(type(agent_output))

        formatter.print_panel(
            f"Agent output type: {type(agent_output)} \n {agent_output}",
            "blue",
        )

        # Convert the output to a dictionary if it's a list
        if isinstance(agent_output, list):
            agent_output = json.dumps(agent_output)

        if isinstance(agent_output, str):
            # Convert the string output to dictionary
            output_dict = (
                str_to_dict(agent_output)
                if isinstance(agent_output, str)
                else agent_output
            )

        # Extract the detailed queries from the output
        # Search for the key "detailed_queries" in the output list[dictionary]
        if isinstance(output_dict, list):
            for item in output_dict:
                if "detailed_queries" in item:
                    queries = item["detailed_queries"]
                    break
        else:
            queries = output_dict.get("detailed_queries", [])

        print(queries)

        # Log the number of queries generated
        formatter.print_panel(
            f"Generated {len(queries)} queries", "blue"
        )

        print(queries)
        print(type(queries))

        return queries

    def step(self, query: str):
        """
        Execute a single research step with maximum parallelism.

        Args:
            query (str): The research query to process

        Returns:
            Formatted conversation history
        """
        try:
            # Get all the queries to process
            queries = self.get_queries(query)

            print(queries)

            # Submit all queries for concurrent processing
            futures = []
            for q in queries:
                future = self.executor.submit(exa_search, q)
                futures.append((q, future))

            # Process results as they complete
            for q, future in futures:
                try:
                    # Get search results only
                    results = future.result()

                    # Add search results to conversation
                    self.conversation.add(
                        role="User",
                        content=f"Search results for {q}: \n {results}",
                    )

                except Exception as e:
                    # Handle any errors in the thread
                    error_msg = (
                        f"Error processing query '{q}': {str(e)}"
                    )
                    console.print(f"[bold red]{error_msg}[/bold red]")
                    self.conversation.add(
                        role="System",
                        content=error_msg,
                    )

            # Generate final comprehensive analysis after all searches are complete
            try:
                final_summary = self.summarization_agent.run(
                    f"Please generate a comprehensive 4,000-word report analyzing the following content: {self.conversation.get_str()}"
                )

                self.conversation.add(
                    role=self.summarization_agent.agent_name,
                    content=final_summary,
                )
            except Exception as e:
                error_msg = (
                    f"Error generating final summary: {str(e)}"
                )
                console.print(f"[bold red]{error_msg}[/bold red]")
                self.conversation.add(
                    role="System",
                    content=error_msg,
                )

            # Return formatted output
            result = history_output_formatter(
                self.conversation, type=self.output_type
            )

            # If output type is JSON, ensure it's properly formatted
            if self.output_type.lower() == "json":
                try:
                    import json

                    if isinstance(result, str):
                        # Try to parse and reformat for pretty printing
                        parsed = json.loads(result)
                        return json.dumps(
                            parsed, indent=2, ensure_ascii=False
                        )
                except (json.JSONDecodeError, TypeError):
                    # If parsing fails, return as-is
                    pass

            return result

        except Exception as e:
            error_msg = f"Critical error in step execution: {str(e)}"
            console.print(f"[bold red]{error_msg}[/bold red]")
            return (
                {"error": error_msg}
                if self.output_type.lower() == "json"
                else error_msg
            )

    def run(self, task: str):
        return self.step(task)

    def batched_run(self, tasks: List[str]):
        """
        Execute a list of research tasks in parallel.

        Args:
            tasks (List[str]): A list of research tasks to execute

        Returns:
            List[str]: A list of formatted conversation histories
        """
        futures = []
        for task in tasks:
            future = self.executor.submit(self.step, task)
            futures.append((task, future))


# Example usage
# if __name__ == "__main__":
#     try:
#         swarm = DeepResearchSwarm(
#             output_type="json",
#         )
#         result = swarm.step(
#             "What is the active tariff situation with mexico? Only create 2 queries"
#         )

#         # Parse and display results in rich format with markdown export
#         swarm.parse_and_display_results(result, export_markdown=True)

#     except Exception as e:
#         print(f"Error running deep research swarm: {str(e)}")
#         import traceback
#         traceback.print_exc()
