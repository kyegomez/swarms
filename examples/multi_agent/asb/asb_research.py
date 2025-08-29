import orjson
from dotenv import load_dotenv

from swarms.structs.auto_swarm_builder import AutoSwarmBuilder

load_dotenv()

swarm = AutoSwarmBuilder(
    name="My Swarm",
    description="My Swarm Description",
    verbose=True,
    max_loops=1,
    return_agents=True,
)

result = swarm.run(
    task="Build a swarm to write a research paper on the topic of AI"
)

print(orjson.dumps(result, option=orjson.OPT_INDENT_2).decode())
