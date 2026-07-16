r"""
Per-node retry & fallback annotations
=====================================

AgentRearrange flows can declare error handling directly on a node
instead of wrapping the whole run in try/except:

    "A -> B!3 -> C"      B retries up to 3 extra times on failure
    "A -> B!3>D -> C"    B retries 3 times, then falls back to D
    "A -> B?D -> C"      B routes to D on the first failure

Rules:
- `!`, `>`, and `?` are reserved characters — they cannot appear in
  agent names used inside a flow.
- Fallback agents must be registered in `agents`; unknown names fail
  in validate_flow / set_custom_flow instead of mid-run.
- The fallback agent runs once. If it also fails, the error
  propagates exactly like an unannotated node's error.
- Annotations work in parallel groups too: "A -> B!2>D, C -> E".

This example builds a research pipeline where the primary analyst is
backed by a cheaper fallback model, so a provider outage on the
primary does not kill the run.
"""

from swarms import Agent, AgentRearrange

MODEL = "gpt-4o-mini"


def _agent(name: str, prompt: str, model: str = MODEL) -> Agent:
    return Agent(
        agent_name=name,
        system_prompt=prompt,
        model_name=model,
        max_loops=1,
        verbose=False,
        persistent_memory=False,
    )


researcher = _agent(
    "Researcher",
    "Collect the key facts on the given topic in 5 bullet points.",
)
analyst = _agent(
    "Analyst",
    "Analyze the research bullets and extract the 3 most important "
    "implications. Be concise.",
)
backup_analyst = _agent(
    "Backup_Analyst",
    "You are the backup analyst. Analyze the research bullets and "
    "extract the 3 most important implications. Be concise.",
    model="gpt-4o-mini",
)
writer = _agent(
    "Writer",
    "Turn the analysis into a short executive summary (<=120 words).",
)

# Analyst retries twice on transient failures, then hands the exact
# same context to Backup_Analyst. Writer never knows the difference.
pipeline = AgentRearrange(
    agents=[researcher, analyst, backup_analyst, writer],
    flow="Researcher -> Analyst!2>Backup_Analyst -> Writer",
    max_loops=1,
    output_type="final",
)

# Inspect the resolved plan without spending tokens.
pipeline.explain()

if __name__ == "__main__":
    result = pipeline.run(
        "The impact of on-device LLMs on cloud inference demand."
    )
    print(result)
