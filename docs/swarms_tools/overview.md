# Swarms Tools Overview

Swarms agents can call external functions, API wrappers, and structured utilities through the tool system. A tool is any callable that gives an agent a bounded capability: search the web, inspect a financial data source, post to a social platform, run a script, query a database, or transform a file. This section explains how to choose, configure, and operate those tools without turning an agent into an uncontrolled script runner.

Use this page as the entry point for the `swarms_tools` navigation section. The lower-level `BaseTool` reference explains schema conversion and execution internals. These guides focus on practical operating patterns: which tool belongs in an agent, what credentials it needs, how to keep outputs auditable, and how to verify that the agent actually used the tool result.

## When to Use Tools

Add a tool when the agent needs information or an action that is not already in the prompt. Common cases include:

- Fresh data such as web pages, financial quotes, social posts, tickets, or documentation.
- Deterministic calculations that should not be left to model reasoning alone.
- External side effects such as publishing, notification, or database writes.
- Local automation such as file conversion, browser interaction, or command execution.

Do not add a tool just because it is available. Every tool expands the agent's authority. Prefer the smallest callable that returns the exact information required for the next decision.

## Basic Function Tool

The simplest tool is a typed Python function with a clear docstring. Type hints and docstrings help the schema generator expose the function safely to the model.

```python
from swarms import Agent


def summarize_ticket(ticket_id: str, include_comments: bool = True) -> dict:
    """Return a normalized support-ticket summary for an internal ticket ID."""
    return {
        "ticket_id": ticket_id,
        "status": "open",
        "priority": "high",
        "summary": "Customer cannot complete onboarding.",
        "comments_included": include_comments,
    }


agent = Agent(
    agent_name="Support Operations Agent",
    system_prompt="Summarize support tickets and recommend the next operational step.",
    tools=[summarize_ticket],
    max_loops=1,
)

result = agent.run("Summarize ticket SUP-1042 and identify the next owner.")
print(result)
```

Keep the callable narrow. A `summarize_ticket` tool is easier to validate than a generic `run_any_support_action` tool.

## Tool Selection Checklist

Before wiring a tool into an agent, answer these questions:

| Question | Why it matters |
| --- | --- |
| What action or data does the tool expose? | Prevents hidden broad authority. |
| Is the tool read-only or write-capable? | Write-capable tools need stronger prompts and confirmation gates. |
| Which environment variables are required? | Makes deployments reproducible. |
| What should happen on timeout or partial failure? | Prevents the agent from hallucinating missing results. |
| What output shape should the agent expect? | Keeps downstream reasoning stable. |
| How will you verify the result? | Tool calls should leave logs, return metadata, or produce testable artifacts. |

## Recommended Structure

For production agents, separate tool configuration from agent orchestration:

```python
import os
from swarms import Agent


def get_customer_profile(customer_id: str) -> dict:
    """Fetch a customer profile from the configured customer API."""
    api_base = os.environ["CUSTOMER_API_BASE_URL"]
    # Replace this example with a real client call in production.
    return {"customer_id": customer_id, "source": api_base, "plan": "team"}


def build_agent() -> Agent:
    return Agent(
        agent_name="Customer Success Agent",
        system_prompt=(
            "Use tools only when customer data is needed. "
            "Cite the returned source fields in the final answer."
        ),
        tools=[get_customer_profile],
        max_loops=2,
    )
```

This pattern keeps secrets in the environment, makes the tool easy to test independently, and gives the agent a specific instruction for how to use the returned evidence.

## Environment Variables

Tools often need credentials. Use environment variables and fail fast when a required value is missing.

```python
import os


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value
```

Recommended conventions:

- Use one variable per provider, such as `EXA_API_KEY`, `FIRECRAWL_API_KEY`, `TWITTER_BEARER_TOKEN`, or `FINANCIAL_DATA_API_KEY`.
- Do not hard-code API keys in examples, prompts, notebooks, or committed files.
- Return a clear error when a credential is missing instead of silently returning fake data.
- In write-capable tools, include the target account, workspace, or project ID in the returned metadata.

## Read-Only vs Write-Capable Tools

Read-only tools retrieve information. They can usually run autonomously if they are rate-limited and their outputs are checked. Write-capable tools create, update, delete, publish, or spend resources. They need additional safeguards:

- Validate the target identifier before executing.
- Make the tool idempotent where possible.
- Return the created resource URL or ID.
- Log enough metadata to audit what happened.
- Consider adding a separate dry-run function for previews.

For example, a social-media agent should use a read-only search tool for research and a separate posting tool for publication. The posting tool should require the exact text, account, and destination.

## Tool Result Design

Return structured data instead of raw strings whenever possible.

```python
def get_page_status(url: str) -> dict:
    """Check whether a page is reachable and return basic response metadata."""
    return {
        "url": url,
        "ok": True,
        "status_code": 200,
        "checked_at": "2026-01-01T00:00:00Z",
    }
```

Structured outputs let the agent distinguish facts from commentary, and they make tests easier to write.

## Related Guides

- [Finance Tools](finance.md): build agents that retrieve market data, run portfolio checks, and separate calculations from recommendations.
- [Search Tools](search.md): combine web search, page extraction, and source-aware synthesis.
- [Twitter Tools](twitter.md): design social-media research and publishing workflows with account safety.
- [BaseTool Reference](../swarms/tools/base_tool.md): inspect schema conversion, validation, and execution utilities.

