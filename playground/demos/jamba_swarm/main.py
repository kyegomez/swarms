# Description: Main file for the Jamba Swarm.
from swarms.utils.loguru_logger import logger
import json
from typing import List

from dotenv import load_dotenv

from swarms import Agent, MixtureOfAgents, OpenAIChat
from jamba_swarm.prompts import BOSS_PLANNER, BOSS_CREATOR
from jamba_swarm.api_schemas import JambaSwarmResponse
from swarms.utils.parse_code import extract_code_from_markdown


load_dotenv()

# Model

model = OpenAIChat()


# Name, system prompt,
def create_and_execute_swarm(
    name: List[str], system_prompt: List[str], task: str
):
    """
    Creates and executes a swarm of agents for the given task.

    Args:
        name (List[str]): A list of names for the agents.
        system_prompt (List[str]): A list of system prompts for the agents.
        task (str): The description of the task for the swarm.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        List[Agent]: A list of agents in the swarm.

    """
    agents = []
    for name, prompt in zip(name, system_prompt):
        agent = Agent(
            agent_name=name,
            system_prompt=prompt,
            agent_description="Generates a spec of agents for the problem at hand.",
            llm=model,
            max_loops=1,
            autosave=True,
            dynamic_temperature_enabled=True,
            dashboard=False,
            verbose=True,
            streaming_on=True,
            # interactive=True, # Set to False to disable interactive mode
            saved_state_path=f"{name}_agent.json",
            # tools=[calculate_profit, generate_report],
            # docs_folder="docs",
            # pdf_path="docs/accounting_agent.pdf",
            # tools=[browser_automation],
        )
        agents.append(agent)

    # MoA
    moa = MixtureOfAgents(
        agents=agents, description=task, final_agent=name[0]
    )

    out = moa.run(
        task,
    )
    print(out)
    return out


# Initialize the agent
planning_agent = Agent(
    agent_name="Boss Director",
    system_prompt=BOSS_PLANNER,
    agent_description="Generates a spec of agents for the problem at hand.",
    llm=model,
    max_loops=1,
    autosave=True,
    dynamic_temperature_enabled=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    # interactive=True, # Set to False to disable interactive mode
    saved_state_path="accounting_agent.json",
    # tools=[calculate_profit, generate_report],
    # docs_folder="docs",
    # pdf_path="docs/accounting_agent.pdf",
    # tools=[browser_automation],
)


# Boss Agent creator
boss_agent_creator = Agent(
    agent_name="Boss Agent Creator",
    system_prompt=BOSS_CREATOR,
    agent_description="Generates a spec of agents for the problem at hand.",
    llm=model,
    max_loops=1,
    autosave=True,
    dynamic_temperature_enabled=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    # interactive=True, # Set to False to disable interactive mode
    saved_state_path="boss_director_agent.json",
    # tools=[calculate_profit, generate_report],
    # docs_folder="docs",
    # pdf_path="docs/accounting_agent.pdf",
    # tools=[create_and_execute_swarm],
)


def parse_agents(json_data):
    if not json_data:
        raise ValueError("Input JSON data is None or empty")

    parsed_data = json.loads(json_data)
    names = []
    system_prompts = []

    for agent in parsed_data["agents"]:
        names.append(agent["agent_name"])
        system_prompts.append(agent["system_prompt"])

    return names, system_prompts


class JambaSwarm:
    def __init__(self, planning_agent, boss_agent_creator):
        self.planning_agent = planning_agent
        self.boss_agent_creator = boss_agent_creator

    def run(self, task: str = None):
        # Planning agent
        logger.info(f"Making plan for the task: {task}")
        out = self.planning_agent.run(task)

        # Boss agent
        logger.info("Running boss agent creator with memory.")
        agents = self.boss_agent_creator.run(out)
        # print(f"Agents: {agents}")
        agents = extract_code_from_markdown(agents)
        logger.info(f"Output from boss agent creator: {agents}")

        # Debugging output
        logger.debug(f"Output from boss agent creator: {agents}")

        # Check if agents is None
        if agents is None:
            raise ValueError("The boss agent creator returned None")

        # Parse the JSON input and output the list of agent names and system prompts
        names, system_prompts = parse_agents(agents)

        # Call the function with parsed data
        response = create_and_execute_swarm(names, system_prompts, task)

        # Create and execute swarm
        log = JambaSwarmResponse(
            task=task,
            plan=out,
            agents=agents,
            response=response,
        )

        return log.json()


swarm = JambaSwarm(planning_agent, boss_agent_creator)

# Run the swarm
swarm.run("Create a swarm of agents for sales")
