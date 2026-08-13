"""
Example 13 — MCP tools inside a multi-agent workflow.

Every other example in this folder is a single agent. This one wires MCP
servers into a ``SequentialWorkflow``, which is the more interesting case for
a multi-agent framework: each agent gets *only* the server it needs, and the
pipeline hands findings down the chain.

    Researcher : DeepWiki   (https://mcp.deepwiki.com/mcp)   — repo Q&A
    Librarian  : Context7   (https://mcp.context7.com/mcp)   — current library docs
    Reporter   : no tools                                    — synthesis only

    Auth : none for either server (verified 2026-08-12)

**Why split the tools per agent instead of giving one agent all of them?**

- *Focus.* A model choosing among many tools for a narrow job picks wrong more
  often than one choosing among two. Each agent here sees a small surface.
- *Context.* Tool schemas occupy the window on every call. Loading Context7's
  schemas into the agent that only reads a repo is pure overhead.
- *Attribution.* When the output is wrong you can tell which stage produced it.
- *Cost.* The reporter needs no tools at all, so it never pays for them.

The stages are genuinely dependent — the librarian looks up whatever
dependencies the researcher found — which is why this is a chain. When stages
*don't* depend on each other, swap ``SequentialWorkflow`` for
``ConcurrentWorkflow`` and they run in parallel; the constructor call is
otherwise identical.

Run:
    export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY, etc.
    python examples/mcp/agents/13_mcp_sequential_workflow.py
"""

from swarms import Agent, SequentialWorkflow

MODEL = "gpt-5.4"

# --- Stage 1: understand the repository (DeepWiki only) -------------------
researcher = Agent(
    agent_name="Repo-Researcher",
    agent_description="Explains a repository's architecture using DeepWiki.",
    system_prompt=(
        "You map codebases. Use DeepWiki to establish what a repository "
        "actually does before describing it. Report the module layout, the "
        "entry points, and — most important for the next stage — the "
        "third-party libraries it depends on, named exactly. Be concrete "
        "about module and package names; a downstream agent cannot look up "
        "something you described only in prose."
    ),
    model_name=MODEL,
    mcp_url="https://mcp.deepwiki.com/mcp",
    max_loops=2,
)

# --- Stage 2: check those dependencies' current docs (Context7 only) ------
librarian = Agent(
    agent_name="Docs-Librarian",
    agent_description="Checks current library documentation via Context7.",
    system_prompt=(
        "You verify how libraries are *currently* meant to be used. Take the "
        "dependencies identified for you and look each one up — resolve the "
        "library id, then fetch its docs. Report the current recommended API "
        "for each, and flag anything deprecated or superseded, since that is "
        "the whole point of checking live docs rather than trusting memory. "
        "If a library cannot be found, say so and move on rather than "
        "inventing its API."
    ),
    model_name=MODEL,
    mcp_url="https://mcp.context7.com/mcp",
    max_loops=3,
)

# --- Stage 3: synthesis (no tools -- nothing left to look up) -------------
reporter = Agent(
    agent_name="Report-Writer",
    agent_description="Turns research and docs findings into a brief.",
    system_prompt=(
        "You write engineering briefs for a technical lead who has five "
        "minutes. Open with the single most important conclusion, then the "
        "supporting detail. Preserve every concrete finding you were handed — "
        "package names, versions, deprecations — and invent none. Where the "
        "earlier stages disagreed or hedged, surface that rather than "
        "smoothing it into false confidence. End with specific next actions."
    ),
    model_name=MODEL,
    max_loops=1,
)

workflow = SequentialWorkflow(
    agents=[researcher, librarian, reporter],
    max_loops=1,
)

if __name__ == "__main__":
    result = workflow.run(
        "Review the modelcontextprotocol/python-sdk repository: map its "
        "architecture, identify its main third-party dependencies, check "
        "whether the APIs it relies on are current or deprecated, and write "
        "a brief for the maintainers."
    )
    print(result)
