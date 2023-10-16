from swarms.models import OpenAIChat

from swarms.swarms import GodMode
from swarms.workers.worker import Worker

llm = OpenAIChat(model_name="gpt-4", openai_api_key="api-key", temperature=0.5)

worker1 = Worker(
    llm=llm,
    ai_name="Bumble Bee",
    ai_role="Worker in a swarm",
    external_tools=None,
    human_in_the_loop=False,
    temperature=0.5,
)
worker2 = Worker(
    llm=llm,
    ai_name="Optimus Prime",
    ai_role="Worker in a swarm",
    external_tools=None,
    human_in_the_loop=False,
    temperature=0.5,
)
worker3 = Worker(
    llm=llm,
    ai_name="Megatron",
    ai_role="Worker in a swarm",
    external_tools=None,
    human_in_the_loop=False,
    temperature=0.5,
)
# Usage
agents = [worker1, worker2, worker3]

god_mode = GodMode(agents)

task = "What are the biggest risks facing humanity?"

god_mode.print_responses(task)
