from swarms.structs.agent import Agent

agent = Agent(
    model_name="gpt-4.1",
    max_loops=1,
    stream=True,
)

agent.run("Tell me a short story about a robot learning to paint.")
