from swarms.structs.auto_swarm_builder import AutoSwarmBuilder
from dotenv import load_dotenv

load_dotenv()

swarm = AutoSwarmBuilder(
    name="My Swarm",
    description="My Swarm Description",
    verbose=True,
    max_loops=1,
)

result = swarm.run(
    task="Build a swarm to write a research paper on the topic of AI"
)

print(result)
