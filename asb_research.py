import orjson

from swarms import AutoSwarmBuilder


swarm = AutoSwarmBuilder(
    name="My Swarm",
    description="My Swarm Description",
    verbose=True,
    max_loops=1,
    execution_type="return-agents",
    model_name="gpt-4.1",
)

result = swarm.run(
    task="Build a swarm to write a research paper on the topic of AI"
)

print(orjson.dumps(result, option=orjson.OPT_INDENT_2).decode())
