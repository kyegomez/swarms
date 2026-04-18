import os
from dotenv import load_dotenv
from swarms import Agent, HierarchicalSwarm
from swarm_models import OpenAIChat

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAIChat(
    openai_api_key=api_key,
    model_name="gpt-4o",
    temperature=0.1,
)

director = Agent(
    agent_name="Director",
    system_prompt="You are the director. You break down tasks and coordinate your team.",
    llm=llm,
    max_loops=1,
)

researcher = Agent(
    agent_name="Researcher",
    system_prompt="You are an expert researcher. You gather and analyze information.",
    llm=llm,
    max_loops=1,
)

writer = Agent(
    agent_name="Writer",
    system_prompt="You are an expert writer. You synthesize information into clear reports.",
    llm=llm,
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
