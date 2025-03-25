import asyncio
import concurrent.futures
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import aiohttp
from dotenv import load_dotenv
from rich.console import Console

from swarms.agents.reasoning_duo import ReasoningDuo
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.any_to_str import any_to_str
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


def format_exa_results(json_data: Dict[str, Any]) -> str:
    """Formats Exa.ai search results into structured text"""
    if "error" in json_data:
        return f"### Error\n{json_data['error']}\n"

    # Pre-allocate formatted_text list with initial capacity
    formatted_text = []

    # Extract search metadata
    search_params = json_data.get("effectiveFilters", {})
    query = search_params.get("query", "General web search")
    formatted_text.append(
        f"### Exa Search Results for: '{query}'\n\n---\n"
    )

    # Process results
    results = json_data.get("results", [])

    if not results:
        formatted_text.append("No results found.\n")
        return "".join(formatted_text)

    def process_result(
        result: Dict[str, Any], index: int
    ) -> List[str]:
        """Process a single result in a thread-safe manner"""
        title = result.get("title", "No title")
        url = result.get("url", result.get("id", "No URL"))
        published_date = result.get("publishedDate", "")

        # Handle highlights efficiently
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

        return [
            f"{index}. **{title}**\n",
            f"   - URL: {url}\n",
            f"   - Published: {published_date.split('T')[0] if published_date else 'Date unknown'}\n",
            f"   - Key Points:\n      {highlight_text}\n\n",
        ]

    # Process results concurrently
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_result = {
            executor.submit(process_result, result, i + 1): i
            for i, result in enumerate(results)
        }

        # Collect results in order
        processed_results = [None] * len(results)
        for future in as_completed(future_to_result):
            idx = future_to_result[future]
            try:
                processed_results[idx] = future.result()
            except Exception as e:
                console.print(
                    f"[bold red]Error processing result {idx + 1}: {str(e)}[/bold red]"
                )
                processed_results[idx] = [
                    f"Error processing result {idx + 1}: {str(e)}\n"
                ]

    # Extend formatted text with processed results in correct order
    for result_text in processed_results:
        formatted_text.extend(result_text)

    return "".join(formatted_text)


async def _async_exa_search(
    query: str, **kwargs: Any
) -> Dict[str, Any]:
    """Asynchronous helper function for Exa.ai API requests"""
    api_url = "https://api.exa.ai/search"
    headers = {
        "x-api-key": os.getenv("EXA_API_KEY"),
        "Content-Type": "application/json",
    }

    payload = {
        "query": query,
        "useAutoprompt": True,
        "numResults": kwargs.get("num_results", 10),
        "contents": {
            "text": True,
            "highlights": {"numSentences": 2},
        },
        **kwargs,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                api_url, json=payload, headers=headers
            ) as response:
                if response.status != 200:
                    return {
                        "error": f"HTTP {response.status}: {await response.text()}"
                    }
                return await response.json()
    except Exception as e:
        return {"error": str(e)}


def exa_search(query: str, **kwargs: Any) -> str:
    """Performs web search using Exa.ai API with concurrent processing"""
    try:
        # Run async search in the event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response_json = loop.run_until_complete(
                _async_exa_search(query, **kwargs)
            )
        finally:
            loop.close()

        # Format results concurrently
        formatted_text = format_exa_results(response_json)

        return formatted_text

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        console.print(f"[bold red]{error_msg}[/bold red]")
        return error_msg


# Define the research tools schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_topic",
            "description": "Conduct an in-depth search on a specified topic or subtopic, generating a comprehensive array of highly detailed search queries tailored to the input parameters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "depth": {
                        "type": "integer",
                        "description": "Indicates the level of thoroughness for the search. Values range from 1 to 3, where 1 represents a superficial search and 3 signifies an exploration of the topic.",
                    },
                    "detailed_queries": {
                        "type": "array",
                        "description": "An array of highly specific search queries that are generated based on the input query and the specified depth. Each query should be designed to elicit detailed and relevant information from various sources.",
                        "items": {
                            "type": "string",
                            "description": "Each item in this array should represent a unique search query that targets a specific aspect of the main topic, ensuring a comprehensive exploration of the subject matter.",
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


# Initialize the research agent
research_agent = Agent(
    agent_name="Deep-Research-Agent",
    agent_description="Specialized agent for conducting comprehensive research across multiple domains",
    system_prompt=RESEARCH_AGENT_PROMPT,
    max_loops=1,  # Allow multiple iterations for thorough research
    tools_list_dictionary=tools,
    model_name="gpt-4o-mini",
)


reasoning_duo = ReasoningDuo(
    system_prompt=SUMMARIZATION_AGENT_PROMPT, output_type="string"
)


class DeepResearchSwarm:
    def __init__(
        self,
        name: str = "DeepResearchSwarm",
        description: str = "A swarm that conducts comprehensive research across multiple domains",
        research_agent: Agent = research_agent,
        max_loops: int = 1,
        nice_print: bool = True,
        output_type: str = "json",
        max_workers: int = os.cpu_count()
        * 2,  # Let the system decide optimal thread count
        token_count: bool = False,
        research_model_name: str = "gpt-4o-mini",
    ):
        self.name = name
        self.description = description
        self.research_agent = research_agent
        self.max_loops = max_loops
        self.nice_print = nice_print
        self.output_type = output_type
        self.max_workers = max_workers
        self.research_model_name = research_model_name

        self.reliability_check()
        self.conversation = Conversation(token_count=token_count)

        # Create a persistent ThreadPoolExecutor for the lifetime of the swarm
        # This eliminates thread creation overhead on each query
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
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

        self.conversation.add(
            role=self.research_agent.agent_name, content=agent_output
        )

        # Convert the string output to dictionary
        output_dict = str_to_dict(agent_output)

        # Print the conversation history
        if self.nice_print:
            to_do_list = any_to_str(output_dict)
            formatter.print_panel(to_do_list, "blue")

        # Extract the detailed queries from the output
        if (
            isinstance(output_dict, dict)
            and "detailed_queries" in output_dict
        ):
            queries = output_dict["detailed_queries"]
            formatter.print_panel(
                f"Generated {len(queries)} queries", "blue"
            )
            return queries

        return []

    def _process_query(self, query: str) -> Tuple[str, str]:
        """
        Process a single query with search and reasoning.
        This function is designed to be run in a separate thread.

        Args:
            query (str): The query to process

        Returns:
            Tuple[str, str]: A tuple containing (search_results, reasoning_output)
        """
        # Run the search
        results = exa_search(query)

        # Run the reasoning on the search results
        reasoning_output = reasoning_duo.run(results)

        return (results, reasoning_output)

    def step(self, query: str):
        """
        Execute a single research step with maximum parallelism.

        Args:
            query (str): The research query to process

        Returns:
            Formatted conversation history
        """
        # Get all the queries to process
        queries = self.get_queries(query)

        # Submit all queries for concurrent processing
        # Using a list instead of generator for clearer debugging
        futures = []
        for q in queries:
            future = self.executor.submit(self._process_query, q)
            futures.append((q, future))

        # Process results as they complete (no waiting for slower queries)
        for q, future in futures:
            try:
                # Get results (blocks until this specific future is done)
                results, reasoning_output = future.result()

                # Add search results to conversation
                self.conversation.add(
                    role="User",
                    content=f"Search results for {q}: \n {results}",
                )

                # Add reasoning output to conversation
                self.conversation.add(
                    role=reasoning_duo.agent_name,
                    content=reasoning_output,
                )
            except Exception as e:
                # Handle any errors in the thread
                self.conversation.add(
                    role="System",
                    content=f"Error processing query '{q}': {str(e)}",
                )

        # Once all query processing is complete, generate the final summary
        # This step runs after all queries to ensure it summarizes all results
        final_summary = reasoning_duo.run(
            f"Generate an extensive report of the following content: {self.conversation.get_str()}"
        )

        self.conversation.add(
            role=reasoning_duo.agent_name,
            content=final_summary,
        )

        return history_output_formatter(
            self.conversation, type=self.output_type
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


# # Example usage
# if __name__ == "__main__":
#     swarm = DeepResearchSwarm(
#         output_type="json",
#     )
#     print(
#         swarm.step(
#             "What is the active tarrif situation with mexico? Only create 2 queries"
#         )
#     )
