# Using Swarms with Firecrawl for Web Research Agents

Firecrawl is useful when an agent needs to turn live websites into clean
markdown before reasoning over them. Swarms can use that capability as a normal
Python tool, which makes it a good fit for competitive research, documentation
audits, lead enrichment, product comparison, and internal knowledge-gathering
workflows.

This guide shows how to build a focused web research agent that crawls a small
set of pages with Firecrawl, asks a Swarms agent to extract evidence, and then
turns the result into a concise research brief. The example uses
`crawl_entire_site_firecrawl` from `swarms-tools`, the same Firecrawl helper used
by the examples in the Swarms repository.

## When to Use This Pattern

Use a Firecrawl-backed research agent when the source material lives on public
web pages and you need cleaner text than a raw HTTP request can provide. The
pattern works best for sites with documentation, marketing pages, help centers,
blog posts, changelogs, and product pages.

Avoid using this pattern for private websites unless you have permission to
crawl them. Also avoid using it as a bulk scraper. Keep page limits small during
development, respect the target site's terms, and cache outputs if you need to
rerun the same analysis.

## Prerequisites

- Python 3.10+
- A Firecrawl API key
- An LLM provider key such as `OPENAI_API_KEY`
- `swarms` and `swarms-tools`

Install the required packages:

```bash
pip install -U swarms swarms-tools python-dotenv
```

Create a `.env` file:

```bash
OPENAI_API_KEY="your-openai-api-key"
FIRECRAWL_API_KEY="your-firecrawl-api-key"
WORKSPACE_DIR="agent_workspace"
```

The Firecrawl helper reads `FIRECRAWL_API_KEY` from the environment. Do not put
API keys directly in prompts, source files, notebooks, or logs.

## Build the Research Tool

The raw Firecrawl helper can crawl many pages and return a large markdown
string. For agent use, wrap it with a smaller, task-specific function. This
keeps defaults explicit and gives the agent a stable tool description.

```python
from dotenv import load_dotenv
from swarms import Agent
from swarms_tools import crawl_entire_site_firecrawl

load_dotenv()


def crawl_research_source(url: str, limit: int = 5) -> str:
    """
    Crawl a public website and return markdown content for research.

    Args:
        url: Public website URL to crawl.
        limit: Maximum number of pages to crawl. Keep this small for research.

    Returns:
        Markdown content and metadata from the crawled pages.
    """
    return crawl_entire_site_firecrawl(
        url=url,
        limit=limit,
        formats=["markdown"],
        max_wait_time=240,
        poll_interval=5,
        include_metadata=True,
    )
```

This wrapper narrows the tool surface to the two inputs a research agent usually
needs: a URL and a page limit. The underlying helper validates the URL, submits a
Firecrawl crawl job, polls until completion, and returns formatted page content.

## Create a Firecrawl Research Agent

Now attach the wrapper as a Swarms tool. The system prompt should tell the agent
how to use web evidence and how to report uncertainty. This is important because
crawled pages can be incomplete, out of date, or biased toward marketing copy.

```python
research_agent = Agent(
    agent_name="Firecrawl-Web-Research-Agent",
    agent_description=(
        "Researches public websites with Firecrawl and produces "
        "evidence-backed summaries."
    ),
    model_name="gpt-5.4",
    max_loops=2,
    dynamic_context_window=True,
    tools=[crawl_research_source],
    system_prompt=(
        "You are a careful web research agent. Use the Firecrawl tool when "
        "you need source material from a public website. Extract specific "
        "claims, product details, pricing signals, docs gaps, and update "
        "dates when available. Do not invent citations. If the crawl does "
        "not contain enough evidence, say what is missing and recommend the "
        "next source to inspect."
    ),
)

brief = research_agent.run(
    "Research https://docs.swarms.world with a 4 page limit. "
    "Return a concise brief with: key capabilities, target users, "
    "notable integration points, and unanswered questions."
)

print(brief)
```

For production workflows, keep `max_loops` low until you understand how often
the model calls the crawler. Web crawls are slower and more expensive than
ordinary local tools, so a research agent should gather evidence once, reason
over it, and ask for another crawl only when the missing information is clear.

## Add a Second Agent for Editorial Review

Many web research tasks benefit from two stages: one agent gathers evidence and
another turns it into a stakeholder-ready brief. You can run those stages
explicitly in Python:

```python
editor_agent = Agent(
    agent_name="Research-Brief-Editor",
    agent_description="Turns raw research notes into concise briefs.",
    model_name="gpt-5.4",
    max_loops=1,
    system_prompt=(
        "You are an editor for technical research briefs. Preserve factual "
        "claims from the research notes, remove repetition, and organize the "
        "answer into: Summary, Evidence, Risks, and Next Steps."
    ),
)

final_brief = editor_agent.run(
    "Edit these research notes into a decision-ready brief:\n\n" + brief
)

print(final_brief)
```

This split is easier to debug than a single broad prompt. If the final answer is
weak, you can inspect whether the crawl returned poor source material, the
research agent missed a detail, or the editor compressed the evidence too much.

## Practical Prompt Template

For repeatable research jobs, use a prompt template like this:

```text
Research {url} with a limit of {page_limit} pages.

Goal: {business_question}

Return:
1. Direct answer
2. Evidence from crawled pages
3. Missing information
4. Recommended next source
5. Confidence level
```

This structure prevents the agent from producing a generic website summary when
you actually need a decision. It also creates a natural audit trail: every brief
shows what was found, what was missing, and where the workflow should look next.

## Reliability Tips

- Start with `limit=3` to verify the target site and prompt behavior.
- Increase `max_wait_time` for large documentation sites.
- Ask for markdown only unless your downstream task truly needs HTML.
- Store the raw Firecrawl output when the result will be reviewed later.
- Put rate limits and crawl budgets around any scheduled workflow.
- Keep a human approval step before publishing competitive claims, pricing
  claims, or legal/compliance conclusions.

## Troubleshooting

If the agent reports that the Firecrawl tool failed, first check
`FIRECRAWL_API_KEY`. If the key is set, reduce the crawl `limit` and retry a
known public page. If the crawl succeeds but the brief is vague, update the
research prompt with more specific fields and require the agent to distinguish
between evidence, inference, and missing information.

If the output is too long for the model context window, lower the page limit or
add a preprocessing tool that truncates each page to the title, headings, and
most relevant sections. Firecrawl is best used as an evidence intake layer;
Swarms should do the planning, extraction, critique, and final synthesis.
