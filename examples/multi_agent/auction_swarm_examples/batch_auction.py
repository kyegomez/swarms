from swarms.structs.agent import Agent
from swarms.structs.auction_swarm import AuctionSwarm

coder = Agent(
    agent_name="Coder",
    agent_description="Writes and debugs software",
    system_prompt="You are an expert software engineer.",
    model_name="gpt-5.4",
    max_loops=1,
    print_on=False,
)

translator = Agent(
    agent_name="Translator",
    agent_description="Translates between human languages",
    system_prompt=(
        "You are a professional translator fluent in many languages."
    ),
    model_name="gpt-5.4",
    max_loops=1,
    print_on=False,
)

chef = Agent(
    agent_name="Chef",
    agent_description="Recipes and cooking guidance",
    system_prompt="You are a professional chef and recipe developer.",
    model_name="gpt-5.4",
    max_loops=1,
    print_on=False,
)

historian = Agent(
    agent_name="Historian",
    agent_description="Historical context and analysis",
    system_prompt="You are a historian specializing in world history.",
    model_name="gpt-5.4",
    max_loops=1,
    print_on=False,
)

swarm = AuctionSwarm(
    name="mixed-expertise-auction",
    agents=[coder, translator, chef, historian],
    top_k=1,
    print_on=True,
)

tasks = [
    "Write a Python function that returns the nth Fibonacci number.",
    "Translate 'Where is the nearest train station?' into French.",
    "Give me a simple recipe for a three-ingredient pasta dish.",
    "Briefly explain what caused the fall of the Western Roman Empire.",
]

if __name__ == "__main__":
    results = swarm.batch_run(tasks)
    for task, result in zip(tasks, results):
        print("\n" + "=" * 70)
        print(f"TASK: {task}")
        print("=" * 70)
        print(result)
