from swarms.swarms import DialogueSimulator
from swarms.workers.worker import Worker
from swarms.models import OpenAIChat

llm = OpenAIChat(
    model_name="gpt-4", openai_api_key="api-key", temperature=0.5
)

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

collab = DialogueSimulator(
    [worker1, worker2],
    # DialogueSimulator.select_next_speaker
)

collab.run(
    max_iters=4,
    name="plinus",
    message="how can we enable multi agent collaboration",
)
