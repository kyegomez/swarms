from swarms.agents.i_agent import IterativeReflectiveExpansion

agent = IterativeReflectiveExpansion(
    max_iterations=3,
)

agent.run("What is the 40th prime number?")
