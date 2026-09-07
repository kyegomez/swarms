from dotenv import load_dotenv

from swarms import Agent

load_dotenv()

agent = Agent(
    agent_name="Astra-Agent",
    system_prompt="You are a concise research assistant.",
    model_name="gpt-6-astra",
    max_loops=1,
    top_p=None,
    temperature=None,
    reasoning_effort="low",
    persistent_memory=False,
    streaming_on=True,
)

out = agent.run(
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than "
    "the ball. How much does the ball cost? Answer with the number only."
)

print(out)
print(agent.usage)
