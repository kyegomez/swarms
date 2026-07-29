from dotenv import load_dotenv
from swarms import Agent, HierarchicalSwarm

load_dotenv()

MODEL_NAME = "gpt-4o"

director = Agent(
    agent_name="Director",
    system_prompt="You are the director. You break down tasks and coordinate your team.",
    model_name=MODEL_NAME,
    temperature=0.1,
    max_loops=1,
)

researcher = Agent(
    agent_name="Researcher",
    system_prompt="You are an expert researcher. You gather and analyze information.",
    model_name=MODEL_NAME,
    temperature=0.1,
    max_loops=1,
)

writer = Agent(
    agent_name="Writer",
    system_prompt="You are an expert writer. You synthesize information into clear reports.",
    model_name=MODEL_NAME,
    temperature=0.1,
    max_loops=1,
)

hscf_swarm = HierarchicalSwarm(
    name="Research and Writing Team",
    description="A hierarchical team that researches and writes comprehensive reports.",
    director=director,
    agents=[researcher, writer],
)

if __name__ == "__main__":
    hscf_swarm.run(
        "Research the latest advancements in solid-state batteries and write a summary report."
    )
