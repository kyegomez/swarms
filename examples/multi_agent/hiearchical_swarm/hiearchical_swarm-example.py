import os
from dotenv import load_dotenv

# Swarm imports
from swarms.structs.agent import Agent
from swarms.structs.hiearchical_swarm import (
    HierarchicalSwarm,
    SwarmSpec,
    OrganizationalUnit,
)
from swarms.utils.function_caller_model import OpenAIFunctionCaller

# Load environment variables
load_dotenv()

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

# Create organizational units with the agents
research_unit = OrganizationalUnit(
    name="Research Unit",
    description="Handles research and analysis tasks",
    manager=research_manager,
    members=[data_analyst, research_assistant],
)

development_unit = OrganizationalUnit(
    name="Development Unit",
    description="Handles development and implementation tasks",
    manager=development_manager,
    members=[software_engineer, qa_engineer],
)

# Initialize the director agent
director = OpenAIFunctionCaller(
    model_name="gpt-4o",
    system_prompt=(
        "As the Director of this Hierarchical Agent Swarm, you are responsible for:\n"
        "1. Analyzing tasks and breaking them down into subtasks\n"
        "2. Assigning tasks to appropriate organizational units\n"
        "3. Coordinating communication between units\n"
        "4. Ensuring tasks are completed efficiently and effectively\n"
        "5. Providing feedback and guidance to units as needed\n\n"
        "Your decisions should be based on the capabilities of each unit and the requirements of the task."
    ),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.5,
    base_model=SwarmSpec,
    max_tokens=10000,
)

# Initialize the hierarchical swarm with the organizational units
swarm = HierarchicalSwarm(
    name="Example Hierarchical Swarm",
    description="A hierarchical swarm demonstrating multi-unit collaboration",
    director=director,
    organizational_units=[research_unit, development_unit],
    max_loops=2,  # Allow for feedback and iteration
    output_type="dict",
)

# Example task to run through the swarm
task = """
Develop a comprehensive market analysis for a new AI-powered productivity tool.
The analysis should include:
1. Market research and competitor analysis
2. User needs and pain points
3. Technical feasibility assessment
4. Implementation recommendations
"""

# Run the task through the swarm
result = swarm.run(task)
print("Swarm Results:", result)
