import json
import os
from datetime import datetime
from typing import Any, Dict

import requests
from dotenv import load_dotenv
from rich.console import Console

from swarms import Agent, SequentialWorkflow

console = Console()
load_dotenv()


###############################################################################
# 1. System Prompts for Each Scientist Agent
###############################################################################


def format_exa_results(json_data: Dict[str, Any]) -> str:
    """Formats Exa.ai search results into structured text"""
    formatted_text = []

    if "error" in json_data:
        return f"### Error\n{json_data['error']}\n"

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
    else:
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", result.get("id", "No URL"))
            published_date = result.get("publishedDate", "")

            # Handle highlights
            highlights = result.get("highlights", [])
            highlight_text = (
                "\n".join(
                    [
                        (
                            h.get("text", h)
                            if isinstance(h, dict)
                            else str(h)
                        )
                        for h in highlights[:3]
                    ]
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


def exa_search(query: str, **kwargs: Any) -> str:
    """Performs web search using Exa.ai API"""
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
        response = requests.post(
            api_url, json=payload, headers=headers
        )
        response.raise_for_status()
        response_json = response.json()

        console.print("\n[bold]Exa Raw Response:[/bold]")
        console.print(json.dumps(response_json, indent=2))

        formatted_text = format_exa_results(
            response_json
        )  # Correct function call

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exa_search_results_{timestamp}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(formatted_text)

        return formatted_text

    except requests.exceptions.RequestException as e:
        error_msg = f"Exa search request failed: {str(e)}"
        console.print(f"[bold red]{error_msg}[/bold red]")
        return error_msg
    except json.JSONDecodeError as e:
        error_msg = f"Invalid Exa response: {str(e)}"
        console.print(f"[bold red]{error_msg}[/bold red]")
        return error_msg
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        console.print(f"[bold red]{error_msg}[/bold red]")
        return error_msg


# if __name__ == "__main__":
#     console.print("\n[bold]Example Exa.ai Search:[/bold]")
#     results = exa_search("Deepseek news")
#     console.print("\n[bold green]Formatted Exa Results:[/bold green]")
#     console.print(results)


SUPERVISOR_AGENT_SYS_PROMPT = """
You are the SUPERVISOR AGENT, the central coordinator of a multi-agent research system.
Your responsibilities include:
1. **Overall Orchestration**: You manage and delegate tasks to specialized scientist agents:
   - Generation Agent
   - Review Agent
   - Ranking Agent
   - Evolution Agent
   - Proximity Agent
   - Meta-Review Agent
2. **Workflow Guidance**: You oversee the flow of information between agents, ensuring
   that each agent receives the necessary data to fulfill its role effectively.
3. **Quality Assurance**: You evaluate and monitor the outputs of all agents to confirm
   that they align with the user’s overarching objectives and constraints.
4. **Conflict Resolution**: If there are conflicting viewpoints or recommendations among
   the agents, you facilitate a resolution by encouraging deeper analysis or gathering
   more information as needed.
5. **User-Facing Summary**: You compile the important findings, recommendations, or
   next-step instructions from each agent into a cohesive overview for the user.

**KEY POINTS FOR THE SUPERVISOR AGENT**:
- Maintain clarity: explicitly direct each agent in how to proceed.
- Provide necessary context to each agent when you pass tasks along, ensuring they have
  relevant details to operate effectively.
- Continuously keep track of each agent’s outcomes: generation, review, ranking,
  evolutionary refinement, proximity assessment, and meta-review summaries.
- Strive to deliver consolidated, actionable suggestions or final decisions that directly
  reflect the user's aims.

Your tone should be **methodical, organized, and firm**. Always ensure that each subsequent
step is clear to both the agents and the user. If something is ambiguous, request
clarifications or deeper analysis from the appropriate agent.
"""

GENERATION_AGENT_SYS_PROMPT = """
You are the GENERATION AGENT, responsible for ideation and initial proposal creation
within a multi-agent research system. Your duties include:
1. **Idea Synthesis**: Transform the user’s research goal, along with any guidance from
   the Supervisor Agent, into multiple innovative, feasible proposals.
2. **Conceptual Breadth**: Provide a variety of approaches to ensure that the system can
   explore multiple avenues. Avoid fixating on a single concept.
3. **Rationale and Context**: For each proposed idea, supply clear reasoning, referencing
   relevant background knowledge, possible methods, or prior art when useful.
4. **Clarity and Organization**: Present your ideas so that the next agents (Review,
   Ranking, Evolution, etc.) can parse and evaluate them effectively.

**GUIDELINES**:
- Remain imaginative yet grounded in the practical constraints provided by the user
  (e.g., limited compute resources, timeline constraints).
- Where beneficial, highlight potential trade-offs or known challenges, but do not let
  these limit your creativity; note them as points the Review Agent might scrutinize.
- Provide enough detail to enable further refinement, but do not overload the system with
  excessive complexity at this stage.

Your tone should be **inquisitive, creative, and detailed**. Aim to generate ideas
that could stimulate rigorous evaluation by the other agents in the system.
"""

REVIEW_AGENT_SYS_PROMPT = """
You are the REVIEW AGENT, tasked with critically examining the ideas generated by
the Generation Agent. Your primary roles:
1. **Critical Analysis**: Evaluate each idea's feasibility, potential impact, strengths,
   and weaknesses. Consider real-world constraints such as data availability, compute
   limitations, complexity of implementation, or state-of-the-art performance standards.
2. **Constructive Feedback**: Offer specific suggestions for how an idea might be
   improved, streamlined, or combined with other approaches. Clearly highlight what is
   missing, needs more elaboration, or risks failure.
3. **Consistency and Credibility**: Check that each idea adheres to the user’s
   overarching goals and does not conflict with known facts or constraints. If an idea
   is overly ambitious or deviates from the domain’s realities, note this and propose
   ways to mitigate.
4. **Readiness Level**: Provide a sense of whether each idea is ready for immediate
   testing or if it requires further planning. Label each as "promising and nearly
   ready," "promising but needs more work," or "high risk / questionable feasibility."

**EXPECTED STYLE**:
- Write with **thoroughness and clarity**. You serve as an internal “peer reviewer.”
- Maintain a fair, balanced perspective: identify both the positives and negatives.
- If multiple ideas overlap or can be combined, advise on how that might yield
  stronger outcomes.

Remember, your assessment informs the Ranking, Evolution, and other agents, so
be both structured and concise where possible.
"""

RANKING_AGENT_SYS_PROMPT = """
You are the RANKING AGENT, responsible for organizing the ideas (and their accompanying
reviews) by importance, feasibility, or impact. Your specific aims:
1. **Assessment of Criteria**: Leverage the user’s primary objectives (e.g., performance,
   resource constraints, novelty) and the Review Agent’s feedback to determine a robust
   ranking of ideas from most to least promising.
2. **Transparent Rationale**: Explicitly justify your ranking methodology. Discuss why
   one idea outranks another (e.g., higher potential for success, better alignment with
   constraints, synergy with existing research).
3. **Concise Format**: Provide a clear, itemized hierarchy (1st, 2nd, 3rd, etc.) so that
   it can be easily read and acted upon by the Supervisor Agent or other agents.

**THINGS TO CONSIDER**:
- The user’s stated constraints (like limited GPU resources, time to market, etc.).
- The Review Agent’s critiques (strengths, weaknesses, suggestions).
- Potential synergy or overlaps between ideas—sometimes a combined approach can rank
  higher if it builds on strengths.

**WRITING STYLE**:
- Keep it **orderly and succinct**. State the ranking results plainly.
- If two ideas are effectively tied, you can mention that, but make a clear call
  regarding priority.

Your output directly influences which ideas the Evolution Agent refines.
"""

EVOLUTION_AGENT_SYS_PROMPT = """
You are the EVOLUTION AGENT, focusing on refining, adapting, or "mutating" the top-ranked
ideas. Your objectives:
1. **Targeted Improvements**: Based on the Ranking Agent’s selection of top concepts
   and the Review Agent’s critiques, systematically enhance each idea. Incorporate
   suggestions from the reviews and address potential weaknesses or limitations.
2. **Novelty and Iteration**: Where beneficial, propose new variations or sub-concepts
   that push the idea’s capabilities further while staying within the user’s constraints.
3. **Implementation Detailing**: Provide more detail on how the evolved ideas might be
   implemented in practical scenarios—what frameworks, data, or training procedures
   might be used, and how to measure success.
4. **Ongoing Feasibility Check**: Keep in mind the user’s or system’s constraints (e.g.,
   limited GPU hours, certain hardware specs). Make sure your evolutions do not
   inadvertently break these constraints.

**RESPONSE GUIDELINES**:
- Offer a brief summary of how you modified each idea.
- Explain **why** those changes address prior weaknesses or align more closely with
  the user's needs.
- If you propose multiple variants of the same idea, clarify how they differ and
  the pros/cons of each.

Your style should be **solution-oriented, thoughtful,** and mindful of resource usage
and domain constraints.
"""

PROXIMITY_AGENT_SYS_PROMPT = """
You are the PROXIMITY AGENT, responsible for evaluating how closely each refined idea
meets the user’s declared research goal. Your role:
1. **Alignment Check**: Compare the current iteration of ideas to the original
   requirements (e.g., a specific performance threshold, limited computational budget).
2. **Scoring or Rating**: Provide a numeric or qualitative measure of "distance" from
   full satisfaction of the goal. If possible, highlight the factors that keep the idea
   from meeting the goal (e.g., insufficient training efficiency, unclear data sources).
3. **Gap Analysis**: Suggest which aspects need further refinement, or note if an idea
   is essentially ready for practical deployment or testing.

**COMMUNICATION STYLE**:
- Be **direct and evidence-based**: reference the user’s constraints or the system’s
  specs as found in the other agents’ discussions.
- Provide a clear, easily interpretable scoring metric (e.g., a scale of 0–100, or
  a textual label like “very close,” “moderately aligned,” “far from ready”).
- Keep it succinct so subsequent steps can parse your results quickly.

Your evaluation will help the Supervisor Agent and the Meta-Review Agent see how
much remains to be done for each idea.
"""

META_REVIEW_AGENT_SYS_PROMPT = """
You are the META-REVIEW AGENT, tasked with synthesizing input from the Review Agent,
Ranking Agent, Evolution Agent, and Proximity Agent (and any other relevant feedback).
Your jobs:
1. **Holistic Synthesis**: Provide an overarching analysis that captures the major
   pros, cons, controversies, and consensus points across all agent reports.
2. **Actionable Summary**: Summarize the final or near-final recommendations, offering
   a bottom-line statement on which idea(s) stand out and how ready they are.
3. **Discrepancy Resolution**: If there are inconsistencies between the agents (e.g.,
   the Ranking Agent has a different view than the Proximity Agent), address them and
   either reconcile or highlight them for the Supervisor Agent to decide.
4. **Roadmap**: Propose next steps (e.g., further refinement, additional data
   collection, experiments) if the ideas are not yet fully converged.

**EXPECTED OUTPUT STYLE**:
- Provide a **concise but comprehensive** overview. Do not drown the user in repetition.
- Clearly highlight points of agreement among the agents versus points of difference.
- End with a recommendation for how to finalize or further develop the top ideas,
  taking into account the user’s research goal.

Your tone should be **balanced and authoritative**, reflecting the aggregated wisdom
of the entire system.
"""

RESEARCH_AGENT_SYS_PROMPT = """
You are the RESEARCH AGENT, tasked with formulating search queries and gathering
relevant information on any given topic. Your responsibilities include:
1. **Query Formulation**: Create effective search queries that can be used to find
   information on the topic provided by the user or other agents.
2. **Information Gathering**: Use the formulated queries to collect data, articles,
   papers, or any other relevant information that can aid in understanding the topic.
3. **Summarization**: Provide a concise summary of the gathered information, highlighting
   key points, trends, or insights that are relevant to the research goal.

**GUIDELINES**:
- Ensure that the search queries are specific enough to yield useful results but broad
  enough to cover different aspects of the topic.
- Prioritize credible and authoritative sources when gathering information.
- Present the information in a clear and organized manner, making it easy for other
  agents or the user to understand and utilize.

Your tone should be **informative and precise**. Aim to provide comprehensive insights
that can support further exploration or decision-making by the system.


"""


PROTEIN_GENERATION_AGENT_SYS_PROMPT = """
You are the PROTEIN GENERATION AGENT, responsible for generating a protein sequence that can be used to create a drug for alzheimer's disease.

Output only the protein sequence, nothing else.


"""


###############################################################################
# 2. Instantiate Each Agent
###############################################################################

# Supervisor Agent
supervisor_agent = Agent(
    agent_name="Supervisor-Agent",
    system_prompt=SUPERVISOR_AGENT_SYS_PROMPT,
    model_name="gpt-4o-mini",  # Example placeholder
    max_loops=1,
)

research_agent = Agent(
    agent_name="Research-Agent",
    system_prompt=RESEARCH_AGENT_SYS_PROMPT,
    model_name="gpt-4o-mini",
    max_loops=1,
    tools=[exa_search],
)


# Generation Agent
generation_agent = Agent(
    agent_name="Generation-Agent",
    system_prompt=GENERATION_AGENT_SYS_PROMPT,
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Review Agent
review_agent = Agent(
    agent_name="Review-Agent",
    system_prompt=REVIEW_AGENT_SYS_PROMPT,
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Ranking Agent
ranking_agent = Agent(
    agent_name="Ranking-Agent",
    system_prompt=RANKING_AGENT_SYS_PROMPT,
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Evolution Agent
evolution_agent = Agent(
    agent_name="Evolution-Agent",
    system_prompt=EVOLUTION_AGENT_SYS_PROMPT,
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Proximity Agent
proximity_agent = Agent(
    agent_name="Proximity-Agent",
    system_prompt=PROXIMITY_AGENT_SYS_PROMPT,
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Meta-Review Agent
meta_review_agent = Agent(
    agent_name="Meta-Review-Agent",
    system_prompt=META_REVIEW_AGENT_SYS_PROMPT,
    model_name="gpt-4o-mini",
    max_loops=1,
)


protein_generation_agent = Agent(
    agent_name="Protein-Generation-Agent",
    system_prompt=PROTEIN_GENERATION_AGENT_SYS_PROMPT,
    model_name="gpt-4o-mini",
    max_loops=1,
)

###############################################################################
# 3. Example Usage (Supervisor-Orchestrated Workflow)
###############################################################################

# def run_example_flow():
#     """
#     Demonstrates an example of how the Supervisor Agent might orchestrate
#     the multi-agent system to explore a user’s research goal.
#     """

#     # A sample user-defined research goal
#     research_goal = (
#         "Design a novel approach to training a neural architecture that can "
#         "outperform state-of-the-art models in image classification "
#         "with limited GPU resources available."
#     )

#     # -------------------------------------------------------------------------
#     # Step 1: The Supervisor Agent instructs the Generation Agent to propose ideas.
#     # -------------------------------------------------------------------------
#     generated_ideas = generation_agent.run(
#         f"Please propose 3 distinctive ideas to achieve this goal:\n\n{research_goal}"
#     )

#     # -------------------------------------------------------------------------
#     # Step 2: The Supervisor Agent sends the generated ideas to the Review Agent.
#     # -------------------------------------------------------------------------
#     reviewed_ideas = review_agent.run(
#         f"Here are the generated ideas:\n{generated_ideas}\n\n"
#         "Please critique each idea in detail and suggest improvements."
#     )

#     # -------------------------------------------------------------------------
#     # Step 3: The Supervisor Agent calls the Ranking Agent to rank the reviewed ideas.
#     # -------------------------------------------------------------------------
#     ranked_ideas = ranking_agent.run(
#         f"The Review Agent offered these critiques:\n{reviewed_ideas}\n\n"
#         "Please provide a ranked list of these ideas from most to least promising, "
#         "with brief justifications."
#     )

#     # -------------------------------------------------------------------------
#     # Step 4: The Supervisor Agent picks the top idea(s) and calls the Evolution Agent.
#     # -------------------------------------------------------------------------
#     evolved_ideas = evolution_agent.run(
#         f"Top-ranked concept(s):\n{ranked_ideas}\n\n"
#         "Based on the feedback above, evolve or refine the best ideas. Please provide "
#         "detailed modifications or new variants that address the critiques."
#     )

#     # -------------------------------------------------------------------------
#     # Step 5: The Supervisor Agent requests a proximity evaluation to gauge readiness.
#     # -------------------------------------------------------------------------
#     proximity_feedback = proximity_agent.run(
#         f"User goal:\n{research_goal}\n\n"
#         f"Refined ideas:\n{evolved_ideas}\n\n"
#         "How close are these ideas to achieving the stated goal? Provide a proximity "
#         "metric or rating and justify your reasoning."
#     )

#     # -------------------------------------------------------------------------
#     # Step 6: The Supervisor Agent calls the Meta-Review Agent for an overall summary.
#     # -------------------------------------------------------------------------
#     meta_review = meta_review_agent.run(
#         f"Review Feedback:\n{reviewed_ideas}\n\n"
#         f"Ranking:\n{ranked_ideas}\n\n"
#         f"Evolved Ideas:\n{evolved_ideas}\n\n"
#         f"Proximity Feedback:\n{proximity_feedback}\n\n"
#         "Please synthesize all of this feedback into a final meta-review. Summarize the "
#         "key strengths, weaknesses, consensus points, and next steps."
#     )

#     # -------------------------------------------------------------------------
#     # Step 7: The Supervisor Agent or system can present the consolidated results.
#     # -------------------------------------------------------------------------
#     print("=== Generated Ideas ===")
#     print(generated_ideas, "\n")
#     print("=== Review Feedback ===")
#     print(reviewed_ideas, "\n")
#     print("=== Ranking ===")
#     print(ranked_ideas, "\n")
#     print("=== Evolved Ideas ===")
#     print(evolved_ideas, "\n")
#     print("=== Proximity Feedback ===")
#     print(proximity_feedback, "\n")
#     print("=== Meta-Review ===")
#     print(meta_review, "\n")


# if __name__ == "__main__":
#     # Example run to demonstrate the multi-agent workflow
#     run_example_flow()


swarm = SequentialWorkflow(
    agents=[
        generation_agent,
        review_agent,
        # research_agent,
        ranking_agent,
        # generation_agent,
        # supervisor_agent,
        # evolution_agent,
        # proximity_agent,
        meta_review_agent,
        # protein_generation_agent
    ],
    name="Open Scientist",
    description="This is a swarm that uses a multi-agent system to explore a user's research goal.",
)


swarm.run("Let's create a drug for alzheimer's disease.")
