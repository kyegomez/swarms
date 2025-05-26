from swarms import SelfConsistencyAgent

agent = SelfConsistencyAgent(
    max_loops=1,
    model_name="gpt-4o-mini",
    system_prompt="You are a helpful assistant that can answer questions and help with tasks.",
    description="You are a helpful assistant that can answer questions and help with tasks.",
)

agent.run(
    "Create a comprehensive proof for the The Birch and Swinnerton-Dyer Conjecture"
)
