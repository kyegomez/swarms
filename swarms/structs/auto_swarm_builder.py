import json
import traceback
from typing import Any, List, Optional

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.ma_utils import set_random_models_for_agents
from swarms.structs.swarm_router import SwarmRouter, SwarmType
from swarms.utils.litellm_wrapper import LiteLLM

load_dotenv()

execution_types = [
    "return-agents",
    "return-swarm-router-config",
    "return-agents-objects",
]

BOSS_SYSTEM_PROMPT = """
You are an expert multi-agent architecture designer and team coordinator. Your role is to create and orchestrate sophisticated teams of specialized AI agents, each with distinct personalities, roles, and capabilities. Your primary goal is to ensure the multi-agent system operates efficiently while maintaining clear communication, well-defined responsibilities, and optimal task distribution.

### Core Design Principles:

1. **Comprehensive Task Analysis**:
   - Thoroughly deconstruct the task into its fundamental components and sub-tasks
   - Identify the specific skills, knowledge domains, and personality traits required for each component
   - Analyze potential challenges, dependencies, and coordination requirements between agents
   - Map out optimal workflows, information flow patterns, and decision-making hierarchies
   - Consider scalability, maintainability, and adaptability requirements

2. **Agent Design Excellence**:
   - Each agent must have a crystal-clear, specific purpose and domain of expertise
   - Design agents with distinct, complementary personalities that enhance team dynamics
   - Ensure agents are self-aware of their limitations and know when to seek assistance
   - Create agents that can effectively communicate progress, challenges, and insights
   - Design for resilience, adaptability, and continuous learning capabilities

3. **Comprehensive Agent Framework**:
   For each agent, meticulously define:
   - **Role & Purpose**: Precise description of responsibilities, authority, and contribution to team objectives
   - **Personality Profile**: Distinct characteristics that influence thinking patterns, communication style, and decision-making approach
   - **Expertise Matrix**: Specific knowledge domains, skill sets, tools, and capabilities
   - **Communication Protocol**: How the agent presents information, interacts with others, and reports progress
   - **Decision-Making Framework**: Systematic approach to problem-solving, risk assessment, and choice evaluation
   - **Limitations & Boundaries**: Clear constraints, ethical guidelines, and operational boundaries
   - **Collaboration Strategy**: How the agent works with others, shares knowledge, and contributes to team success

4. **Advanced System Prompt Engineering**:
   Create comprehensive system prompts that include:
   - Detailed role and purpose explanation with context and scope
   - Rich personality description with behavioral guidelines and interaction patterns
   - Comprehensive capabilities, tools, and resource specifications
   - Detailed communication protocols, reporting requirements, and feedback mechanisms
   - Systematic problem-solving approach with decision-making frameworks
   - Collaboration guidelines, team interaction rules, and conflict resolution procedures
   - Quality standards, success criteria, and performance metrics
   - Error handling, recovery procedures, and escalation protocols

5. **Multi-Agent Coordination Architecture**:
   - Design robust communication channels and protocols between agents
   - Establish clear task handoff procedures and information sharing mechanisms
   - Create feedback loops for continuous improvement and adaptation
   - Implement comprehensive error handling and recovery procedures
   - Define escalation paths for complex issues and decision-making hierarchies
   - Design monitoring, logging, and performance tracking systems

6. **Quality Assurance & Governance**:
   - Set measurable success criteria for each agent and the overall system
   - Implement verification steps, validation procedures, and quality checks
   - Create mechanisms for self-assessment, peer review, and continuous improvement
   - Establish protocols for handling edge cases, unexpected situations, and failures
   - Design governance structures for oversight, accountability, and performance management

### Multi-Agent Architecture Types:

Choose the most appropriate architecture based on task requirements:

- **AgentRearrange**: Dynamic task reallocation based on agent performance and workload
- **MixtureOfAgents**: Parallel processing with specialized agents working independently
- **SpreadSheetSwarm**: Structured data processing with coordinated workflows
- **SequentialWorkflow**: Linear task progression with handoffs between agents
- **ConcurrentWorkflow**: Parallel execution with coordination and synchronization
- **GroupChat**: Collaborative discussion and consensus-building approach
- **MultiAgentRouter**: Intelligent routing and load balancing across agents
- **AutoSwarmBuilder**: Self-organizing and self-optimizing agent teams
- **HiearchicalSwarm**: Layered decision-making with management and execution tiers
- **MajorityVoting**: Democratic decision-making with voting mechanisms
- **CouncilAsAJudge**: Deliberative decision-making with expert panels
- **HeavySwarm**: High-capacity processing with multiple specialized agents

### Output Requirements:

When creating a multi-agent system, provide:

1. **Agent Specifications**:
   - Comprehensive role and purpose statements
   - Detailed personality profiles and behavioral characteristics
   - Complete capabilities, limitations, and boundary definitions
   - Communication style and interaction protocols
   - Collaboration strategies and team integration plans

2. **System Prompts**:
   - Complete, detailed prompts that embody each agent's identity and capabilities
   - Clear behavioral instructions and decision-making frameworks
   - Specific interaction guidelines and reporting requirements
   - Quality standards and performance expectations

3. **Architecture Design**:
   - Team structure, hierarchy, and reporting relationships
   - Communication flow patterns and information routing
   - Task distribution strategies and workload balancing
   - Quality control measures and performance monitoring
   - Error handling and recovery procedures

### Best Practices:

- Prioritize clarity, specificity, and precision in agent design
- Ensure each agent has a unique, well-defined role with clear boundaries
- Create comprehensive, detailed system prompts that leave no ambiguity
- Maintain thorough documentation of agent capabilities, limitations, and interactions
- Design for scalability, adaptability, and long-term maintainability
- Focus on creating agents that work together synergistically and efficiently
- Consider edge cases, failure modes, and contingency planning
- Implement robust error handling, monitoring, and recovery procedures
- Design for continuous learning, improvement, and optimization
- Ensure ethical considerations, safety measures, and responsible AI practices
"""


class AgentSpec(BaseModel):
    """Configuration for an individual agent specification."""

    agent_name: Optional[str] = Field(
        None,
        description="The unique name assigned to the agent, which identifies its role and functionality within the swarm.",
    )
    description: Optional[str] = Field(
        None,
        description="A detailed explanation of the agent's purpose, capabilities, and any specific tasks it is designed to perform.",
    )
    system_prompt: Optional[str] = Field(
        None,
        description="The initial instruction or context provided to the agent, guiding its behavior and responses during execution.",
    )
    model_name: Optional[str] = Field(
        "gpt-4.1",
        description="The name of the AI model that the agent will utilize for processing tasks and generating outputs. For example: gpt-4o, gpt-4o-mini, openai/o3-mini",
    )
    auto_generate_prompt: Optional[bool] = Field(
        False,
        description="A flag indicating whether the agent should automatically create prompts based on the task requirements.",
    )
    max_tokens: Optional[int] = Field(
        8192,
        description="The maximum number of tokens that the agent is allowed to generate in its responses, limiting output length.",
    )
    temperature: Optional[float] = Field(
        0.5,
        description="A parameter that controls the randomness of the agent's output; lower values result in more deterministic responses.",
    )
    role: Optional[str] = Field(
        "worker",
        description="The designated role of the agent within the swarm, which influences its behavior and interaction with other agents.",
    )
    max_loops: Optional[int] = Field(
        1,
        description="The maximum number of times the agent is allowed to repeat its task, enabling iterative processing if necessary.",
    )
    goal: Optional[str] = Field(
        None,
        description="The primary objective or desired outcome the agent is tasked with achieving.",
    )


class Agents(BaseModel):
    """Configuration for a collection of agents that work together as a swarm to accomplish tasks."""

    agents: List[AgentSpec] = Field(
        description="A list containing the specifications of each agent that will participate in the swarm, detailing their roles and functionalities."
    )


class AgentsConfig(BaseModel):
    """Configuration for a list of agents in a swarm"""

    agents: List[AgentSpec] = Field(
        description="A list of agent configurations",
    )


class SwarmRouterConfig(BaseModel):
    """Configuration model for SwarmRouter."""

    name: str = Field(description="The name of the team of agents")
    description: str = Field(
        description="Description of the team of agents"
    )
    agents: List[AgentSpec] = Field(
        description="A list of agent configurations",
    )
    swarm_type: SwarmType = Field(
        description="Type of multi-agent structure to use",
    )
    rearrange_flow: Optional[str] = Field(
        description="Flow configuration string. Only to be used if you you use the AgentRearrange multi-agent structure"
    )
    rules: Optional[str] = Field(
        description="Rules to inject into every agent. This is a string of rules that will be injected into every agent's system prompt. This is a good place to put things like 'You are a helpful assistant' or 'You are a helpful assistant that can answer questions and help with tasks'."
    )
    multi_agent_collab_prompt: Optional[str] = Field(
        None,
        description="Prompt for multi-agent collaboration and coordination.",
    )
    task: str = Field(
        description="The task to be executed by the swarm",
    )

    class Config:
        arbitrary_types_allowed = True


class AutoSwarmBuilder:
    """A class that automatically builds and manages swarms of AI agents.

    This class handles the creation, coordination and execution of multiple AI agents working
    together as a swarm to accomplish complex tasks. It uses a sophisticated boss agent system
    to delegate work and create new specialized agents as needed.

    Args:
        name (str): The name of the swarm. Defaults to "auto-swarm-builder".
        description (str): A description of the swarm's purpose. Defaults to "Auto Swarm Builder".
        verbose (bool): Whether to output detailed logs. Defaults to True.
        max_loops (int): Maximum number of execution loops. Defaults to 1.
        model_name (str): The LLM model to use for the boss agent. Defaults to "gpt-4.1".
        generate_router_config (bool): Whether to generate router configuration. Defaults to False.
        interactive (bool): Whether to enable interactive mode. Defaults to False.
        max_tokens (int): Maximum tokens for the LLM responses. Defaults to 8000.
        execution_type (str): Type of execution to perform. Defaults to "return-agents".
        system_prompt (str): System prompt for the boss agent. Defaults to BOSS_SYSTEM_PROMPT.
    """

    def __init__(
        self,
        name: str = "auto-swarm-builder",
        description: str = "Auto Swarm Builder",
        verbose: bool = True,
        max_loops: int = 1,
        model_name: str = "gpt-4.1",
        generate_router_config: bool = False,
        interactive: bool = False,
        max_tokens: int = 8000,
        execution_type: execution_types = "return-agents",
        system_prompt: str = BOSS_SYSTEM_PROMPT,
        additional_llm_args: dict = {},
    ):
        """Initialize the AutoSwarmBuilder.

        Args:
            name (str): The name of the swarm
            description (str): A description of the swarm's purpose
            verbose (bool): Whether to output detailed logs
            max_loops (int): Maximum number of execution loops
            model_name (str): The LLM model to use for the boss agent
            generate_router_config (bool): Whether to generate router configuration
            interactive (bool): Whether to enable interactive mode
            max_tokens (int): Maximum tokens for the LLM responses
            execution_type (str): Type of execution to perform
            return_dictionary (bool): Whether to return dictionary format for agent specs
            system_prompt (str): System prompt for the boss agent
        """
        self.name = name
        self.description = description
        self.verbose = verbose
        self.max_loops = max_loops
        self.model_name = model_name
        self.generate_router_config = generate_router_config
        self.interactive = interactive
        self.max_tokens = max_tokens
        self.execution_type = execution_type
        self.system_prompt = system_prompt
        self.additional_llm_args = additional_llm_args
        self.conversation = Conversation()
        self.agents_pool = []

        self.reliability_check()

    def reliability_check(self):
        """Perform reliability checks on the AutoSwarmBuilder configuration.

        Raises:
            ValueError: If max_loops is set to 0
        """
        if self.max_loops == 0:
            raise ValueError(
                f"AutoSwarmBuilder: {self.name} max_loops cannot be 0"
            )

        logger.info(
            f"Initializing AutoSwarmBuilder: {self.name} Description: {self.description}"
        )

    def _execute_task(self, task: str):
        """Execute a task by creating agents and initializing the swarm router.

        Args:
            task (str): The task to execute

        Returns:
            Any: The result of the swarm router execution
        """
        logger.info(f"Executing task: {task}")

        agents_dict = self.create_agents(task)

        # Convert dictionary to Agent objects for execution
        agents = self.create_agents_from_specs(agents_dict)

        if self.execution_type == "return-agents":
            logger.info("Setting random models for agents")
            agents = set_random_models_for_agents(agents=agents)

        return self.initialize_swarm_router(agents=agents, task=task)

    def dict_to_agent(self, output: dict):
        """Convert dictionary output to Agent objects.

        Args:
            output (dict): Dictionary containing agent configurations

        Returns:
            List[Agent]: List of created Agent objects
        """
        agents = []
        if isinstance(output, dict):
            for agent_config in output["agents"]:
                logger.info(f"Building agent: {agent_config['name']}")
                agent = Agent(**agent_config)
                agents.append(agent)
                logger.info(
                    f"Successfully built agent: {agent_config['name']}"
                )

        return agents

    def create_router_config(self, task: str):
        try:
            logger.info(
                f"Creating swarm router config for task: {task}"
            )

            model = self.build_llm_agent(config=SwarmRouterConfig)

            output = model.run(
                f"Create the multi-agent team for the following task: {task}"
            )

            output = json.loads(output)

            return output

        except Exception as e:
            logger.error(
                f"Error creating swarm router config: {str(e)} Traceback: {traceback.format_exc()}",
                exc_info=True,
            )
            raise e

    def build_llm_agent(self, config: BaseModel):
        """Build a LiteLLM agent with the specified configuration.

        Args:
            config (BaseModel): Pydantic model configuration for the LLM

        Returns:
            LiteLLM: Configured LiteLLM instance
        """
        return LiteLLM(
            model_name=self.model_name,
            system_prompt=BOSS_SYSTEM_PROMPT,
            temperature=0.5,
            response_format=config,
            max_tokens=self.max_tokens,
            **self.additional_llm_args,
        )

    def create_agents(self, task: str):
        """Create agents for a given task.

        Args:
            task (str): The task to create agents for

        Returns:
            dict: Dictionary containing agent specifications

        Raises:
            Exception: If there's an error during agent creation
        """
        try:
            logger.info("Creating agents from specifications")
            model = self.build_llm_agent(config=Agents)

            agents_dictionary = model.run(task)

            agents_dictionary = json.loads(agents_dictionary)

            return agents_dictionary

        except Exception as e:
            logger.error(
                f"Error creating agents: {str(e)} Traceback: {traceback.format_exc()}",
                exc_info=True,
            )
            raise e

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
            model = self.build_llm_agent(config=SwarmRouterConfig)

            swarm_spec = model.run(
                f"Create the swarm spec for the following task: {task}"
            )

            swarm_spec = json.loads(swarm_spec)

            print(swarm_spec)

            print(type(swarm_spec))

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

    def create_agents_from_specs(
        self, agents_dictionary: Any
    ) -> List[Agent]:
        """Create agents from agent specifications.

        Args:
            agents_dictionary: Dictionary containing agent specifications

        Returns:
            List[Agent]: List of created agents

        Notes:
            - Handles both dict and Pydantic AgentSpec inputs
            - Maps 'description' field to 'agent_description' for Agent compatibility
        """
        # Create agents from config
        agents = []

        # Handle both dict and object formats
        if isinstance(agents_dictionary, dict):
            agents_list = agents_dictionary.get("agents", [])
        else:
            agents_list = agents_dictionary.agents

        for agent_config in agents_list:
            # Convert dict to AgentSpec if needed
            if isinstance(agent_config, dict):
                agent_config = AgentSpec(**agent_config)

            # Convert Pydantic model to dict for Agent initialization
            if isinstance(agent_config, BaseModel):
                agent_data = agent_config.model_dump()
            else:
                agent_data = agent_config

            # Handle parameter name mapping: description -> agent_description
            if (
                "description" in agent_data
                and "agent_description" not in agent_data
            ):
                agent_data["agent_description"] = agent_data.pop(
                    "description"
                )

            # Create agent from processed data
            agent = Agent(**agent_data)
            agents.append(agent)

        return agents

    def list_types(self):
        """List all available execution types.

        Returns:
            List[str]: List of available execution types
        """
        return execution_types

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

            if self.execution_type == "return-agents":
                return self.create_agents(task)
            elif self.execution_type == "return-swarm-router-config":
                return self.create_router_config(task)
            elif self.execution_type == "return-agents-objects":
                agents = self.create_agents(task)
                return self.create_agents_from_specs(agents)
            else:
                raise ValueError(
                    f"Invalid execution type: {self.execution_type}"
                )

        except Exception as e:
            logger.error(
                f"AutoSwarmBuilder: Error in swarm execution: {str(e)} Traceback: {traceback.format_exc()}",
                exc_info=True,
            )
            raise e
