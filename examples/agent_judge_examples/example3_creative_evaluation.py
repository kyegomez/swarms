from swarms.agents.agent_judge import AgentJudge

# Initialize the agent judge for creative content evaluation
judge = AgentJudge(
    agent_name="creative-judge",
    model_name="gpt-4",
    max_loops=2,
    evaluation_criteria={
        "creativity": 0.4,
        "originality": 0.3,
        "engagement": 0.2,
        "coherence": 0.1,
    },
)

# Example creative agent output to evaluate
creative_output = "The moon hung like a silver coin in the velvet sky, casting shadows that danced with the wind. Ancient trees whispered secrets to the stars, while time itself seemed to pause in reverence of this magical moment. The world held its breath, waiting for the next chapter of the eternal story."

# Run evaluation with context building
evaluations = judge.run(task=creative_output)
