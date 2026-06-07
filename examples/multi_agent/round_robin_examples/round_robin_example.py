from swarms import Agent, RoundRobinSwarm

# Three specialists. RoundRobinSwarm will visit them in this exact order
# every loop: Optimist -> Skeptic -> Synthesiser -> Optimist -> ...
# Each agent reads the full transcript so far and knows who spoke before
# it and who is up next (injected into the prompt automatically).

optimist = Agent(
    agent_name="Optimist",
    agent_description="Argues for the upside of the proposal.",
    system_prompt=(
        "You are an optimist. Your job is to identify the strongest "
        "reasons the proposal under discussion will succeed. Be specific."
    ),
    model_name="claude-sonnet-4-6",
    max_loops=1,
)

skeptic = Agent(
    agent_name="Skeptic",
    agent_description="Stress-tests claims and surfaces risks.",
    system_prompt=(
        "You are a skeptic. Your job is to challenge the previous "
        "speaker's strongest claim with a concrete failure mode or "
        "missing assumption."
    ),
    model_name="gpt-5.4",
    max_loops=1,
)

synthesiser = Agent(
    agent_name="Synthesiser",
    agent_description="Distils the discussion into a working position.",
    system_prompt=(
        "You are a synthesiser. Reconcile the optimist's case and the "
        "skeptic's pushback into one clear recommendation."
    ),
    model_name="gemini/gemini-2.5-pro",
    max_loops=1,
)

swarm = RoundRobinSwarm(
    name="Decision-Roundtable",
    agents=[optimist, skeptic, synthesiser],
    max_loops=1,  # two full cycles → 6 turns total
    output_type="all",
)

result = swarm.run(
    "Should a 30-person Series A startup build their own LLM "
    "fine-tuning pipeline or rent one from a vendor?"
)

print(result)
