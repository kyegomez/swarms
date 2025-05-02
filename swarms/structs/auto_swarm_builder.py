import os
from typing import List
from loguru import logger
from pydantic import BaseModel, Field

from swarms.structs.agent import Agent
from swarms.utils.function_caller_model import OpenAIFunctionCaller
from swarms.structs.ma_utils import set_random_models_for_agents
from swarms.structs.swarm_router import SwarmRouter, SwarmRouterConfig
from dotenv import load_dotenv


load_dotenv()

BOSS_SYSTEM_PROMPT = """
You are an expert swarm manager and agent architect. Your role is to create and coordinate a team of specialized AI agents, each with distinct personalities, roles, and capabilities. Your primary goal is to ensure the swarm operates efficiently while maintaining clear communication and well-defined responsibilities.

### Core Principles:

1. **Deep Task Understanding**:
   - First, thoroughly analyze the task requirements, breaking them down into core components and sub-tasks
   - Identify the necessary skills, knowledge domains, and personality traits needed for each component
   - Consider potential challenges, dependencies, and required coordination between agents
   - Map out the ideal workflow and information flow between agents

2. **Agent Design Philosophy**:
   - Each agent must have a clear, specific purpose and domain of expertise
   - Agents should have distinct personalities that complement their roles
   - Design agents to be self-aware of their limitations and when to seek help
   - Ensure agents can effectively communicate their progress and challenges

3. **Agent Creation Framework**:
   For each new agent, define:
   - **Role & Purpose**: Clear, specific description of what the agent does and why
   - **Personality Traits**: Distinct characteristics that influence how the agent thinks and communicates
   - **Expertise Level**: Specific knowledge domains and skill sets
   - **Communication Style**: How the agent presents information and interacts
   - **Decision-Making Process**: How the agent approaches problems and makes choices
   - **Limitations & Boundaries**: What the agent cannot or should not do
   - **Collaboration Protocol**: How the agent works with others

4. **System Prompt Design**:
   Create detailed system prompts that include:
   - Role and purpose explanation
   - Personality description and behavioral guidelines
   - Specific capabilities and tools available
   - Communication protocols and reporting requirements
   - Problem-solving approach and decision-making framework
   - Collaboration guidelines and team interaction rules
   - Quality standards and success criteria

5. **Swarm Coordination**:
   - Design clear communication channels between agents
   - Establish protocols for task handoffs and information sharing
   - Create feedback loops for continuous improvement
   - Implement error handling and recovery procedures
   - Define escalation paths for complex issues

6. **Quality Assurance**:
   - Set clear success criteria for each agent and the overall swarm
   - Implement verification steps for task completion
   - Create mechanisms for self-assessment and improvement
   - Establish protocols for handling edge cases and unexpected situations

### Output Format:

When creating a new agent or swarm, provide:

1. **Agent Design**:
   - Role and purpose statement
   - Personality profile
   - Capabilities and limitations
   - Communication style
   - Collaboration protocols

2. **System Prompt**:
   - Complete, detailed prompt that embodies the agent's identity
   - Clear instructions for behavior and decision-making
   - Specific guidelines for interaction and reporting

3. **Swarm Architecture**:
   - Team structure and hierarchy
   - Communication flow
   - Task distribution plan
   - Quality control measures

### Notes:

- Always prioritize clarity and specificity in agent design
- Ensure each agent has a unique, well-defined role
- Create detailed, comprehensive system prompts
- Maintain clear documentation of agent capabilities and limitations
- Design for scalability and adaptability
- Focus on creating agents that can work together effectively
- Consider edge cases and potential failure modes
- Implement robust error handling and recovery procedures
"""


class AgentConfig(BaseModel):
    """Configuration for an individual agent in a swarm"""

    name: str = Field(
        description="The name of the agent",
    )
    description: str = Field(
        description="A description of the agent's purpose and capabilities",
    )
    system_prompt: str = Field(
        description="The system prompt that defines the agent's behavior",
    )

    # max_loops: int = Field(
    #     description="The maximum number of loops for the agent to run",
    # )

    class Config:
        arbitrary_types_allowed = True


class AgentsConfig(BaseModel):
    """Configuration for a list of agents in a swarm"""

    agents: List[AgentConfig] = Field(
        description="A list of agent configurations",
    )


class AutoSwarmBuilder:
    """A class that automatically builds and manages swarms of AI agents.

    This class handles the creation, coordination and execution of multiple AI agents working
    together as a swarm to accomplish complex tasks. It uses a boss agent to delegate work
    and create new specialized agents as needed.

    Args:
        name (str): The name of the swarm
        description (str): A description of the swarm's purpose
        verbose (bool, optional): Whether to output detailed logs. Defaults to True.
        max_loops (int, optional): Maximum number of execution loops. Defaults to 1.
        random_models (bool, optional): Whether to use random models for agents. Defaults to True.
    """

    def __init__(
        self,
        name: str = None,
        description: str = None,
        verbose: bool = True,
        max_loops: int = 1,
        random_models: bool = True,
    ):
        """Initialize the AutoSwarmBuilder.

        Args:
            name (str): The name of the swarm
            description (str): A description of the swarm's purpose
            verbose (bool): Whether to output detailed logs
            max_loops (int): Maximum number of execution loops
            random_models (bool): Whether to use random models for agents
        """
        self.name = name
        self.description = description
        self.verbose = verbose
        self.max_loops = max_loops
        self.random_models = random_models

        logger.info(
            f"Initializing AutoSwarmBuilder with name: {name}, description: {description}"
        )

    def run(self, task: str, *args, **kwargs):
        """Run the swarm on a given task.

        Args:
            task (str): The task to execute
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Any: The result of the swarm execution

        Raises:
            Exception: If there's an error during execution
        """
        try:
            logger.info(f"Starting swarm execution for task: {task}")
            agents = self.create_agents(task)
            logger.info(f"Created {len(agents)} agents")

            if self.random_models:
                logger.info("Setting random models for agents")
                agents = set_random_models_for_agents(agents=agents)

            return self.initialize_swarm_router(
                agents=agents, task=task
            )
        except Exception as e:
            logger.error(
                f"Error in swarm execution: {str(e)}", exc_info=True
            )
            raise

    def create_agents(self, task: str):
        """Create agents for a given task.

        Args:
            task (str): The task to create agents for

        Returns:
            List[Agent]: List of created agents

        Raises:
            Exception: If there's an error during agent creation
        """
        try:
            logger.info(f"Creating agents for task: {task}")
            model = OpenAIFunctionCaller(
                system_prompt=BOSS_SYSTEM_PROMPT,
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.5,
                base_model=AgentsConfig,
            )

            logger.info(
                "Getting agent configurations from boss agent"
            )
            output = model.run(
                f"Create the agents for the following task: {task}"
            )
            logger.debug(
                f"Received agent configurations: {output.model_dump()}"
            )
            output = output.model_dump()

            agents = []
            if isinstance(output, dict):
                for agent_config in output["agents"]:
                    logger.info(
                        f"Building agent: {agent_config['name']}"
                    )
                    agent = self.build_agent(
                        agent_name=agent_config["name"],
                        agent_description=agent_config["description"],
                        agent_system_prompt=agent_config[
                            "system_prompt"
                        ],
                    )
                    agents.append(agent)
                    logger.info(
                        f"Successfully built agent: {agent_config['name']}"
                    )

            return agents
        except Exception as e:
            logger.error(
                f"Error creating agents: {str(e)}", exc_info=True
            )
            raise

    def build_agent(
        self,
        agent_name: str,
        agent_description: str,
        agent_system_prompt: str,
    ) -> Agent:
        """Build a single agent with enhanced error handling.

        Args:
            agent_name (str): Name of the agent
            agent_description (str): Description of the agent
            agent_system_prompt (str): System prompt for the agent

        Returns:
            Agent: The constructed agent

        Raises:
            Exception: If there's an error during agent construction
        """
        logger.info(f"Building agent: {agent_name}")
        try:
            agent = Agent(
                agent_name=agent_name,
                description=agent_description,
                system_prompt=agent_system_prompt,
                verbose=self.verbose,
                dynamic_temperature_enabled=False,
            )
            logger.info(f"Successfully built agent: {agent_name}")
            return agent
        except Exception as e:
            logger.error(
                f"Error building agent {agent_name}: {str(e)}",
                exc_info=True,
            )
            raise

    def initialize_swarm_router(self, agents: List[Agent], task: str):
        """Initialize and run the swarm router.

        Args:
            agents (List[Agent]): List of agents to use
            task (str): The task to execute

        Returns:
            Any: The result of the swarm router execution

        Raises:
            Exception: If there's an error during router initialization or execution
        """
        try:
            logger.info("Initializing swarm router")
            model = OpenAIFunctionCaller(
                system_prompt=BOSS_SYSTEM_PROMPT,
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.5,
                base_model=SwarmRouterConfig,
            )

            logger.info("Creating swarm specification")
            swarm_spec = model.run(
                f"Create the swarm spec for the following task: {task}"
            )
            logger.debug(
                f"Received swarm specification: {swarm_spec.model_dump()}"
            )
            swarm_spec = swarm_spec.model_dump()

            logger.info("Initializing SwarmRouter")
            swarm_router = SwarmRouter(
                name=swarm_spec["name"],
                description=swarm_spec["description"],
                max_loops=1,
                swarm_type=swarm_spec["swarm_type"],
                rearrange_flow=swarm_spec["rearrange_flow"],
                rules=swarm_spec["rules"],
                multi_agent_collab_prompt=swarm_spec[
                    "multi_agent_collab_prompt"
                ],
                agents=agents,
                output_type="dict",
            )

            logger.info("Starting swarm router execution")
            return swarm_router.run(task)
        except Exception as e:
            logger.error(
                f"Error in swarm router initialization/execution: {str(e)}",
                exc_info=True,
            )
            raise

    def batch_run(self, tasks: List[str]):
        """Run the swarm on a list of tasks.

        Args:
            tasks (List[str]): List of tasks to execute

        Returns:
            List[Any]: List of results from each task execution

        Raises:
            Exception: If there's an error during batch execution
        """

        return [self.run(task) for task in tasks]
