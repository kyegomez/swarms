from swarms.structs.agent import Agent
from swarms.structs.swarm_router import SwarmRouter
import json

# Create the agents first
research_manager = Agent(
    agent_name="Research Manager",
    agent_description="Manages research operations and coordinates research tasks",
    system_prompt="You are a research manager responsible for overseeing research projects and coordinating research efforts.",
    model_name="gpt-4o",
)

data_analyst = Agent(
    agent_name="Data Analyst",
    agent_description="Analyzes data and generates insights",
    system_prompt="You are a data analyst specializing in processing and analyzing data to extract meaningful insights.",
    model_name="gpt-4o",
)

research_assistant = Agent(
    agent_name="Research Assistant",
    agent_description="Assists with research tasks and data collection",
    system_prompt="You are a research assistant who helps gather information and support research activities.",
    model_name="gpt-4o",
)

development_manager = Agent(
    agent_name="Development Manager",
    agent_description="Manages development projects and coordinates development tasks",
    system_prompt="You are a development manager responsible for overseeing software development projects and coordinating development efforts.",
    model_name="gpt-4o",
)

software_engineer = Agent(
    agent_name="Software Engineer",
    agent_description="Develops and implements software solutions",
    system_prompt="You are a software engineer specializing in building and implementing software solutions.",
    model_name="gpt-4o",
)

qa_engineer = Agent(
    agent_name="QA Engineer",
    agent_description="Tests and ensures quality of software",
    system_prompt="You are a QA engineer responsible for testing software and ensuring its quality.",
    model_name="gpt-4o",
)

swarm_router = SwarmRouter(
    name="Swarm Router",
    description="A swarm router that routes tasks to the appropriate agents",
    agents=[
        research_manager,
        data_analyst,
        research_assistant,
        development_manager,
        software_engineer,
        qa_engineer,
    ],
    multi_agent_collab_prompt=True,
    swarm_type="MixtureOfAgents",
    output_type="dict",
)

output = swarm_router.run(
    task="Write a research paper on the impact of AI on the future of work"
)

with open("output.json", "w") as f:
    json.dump(output, f)
