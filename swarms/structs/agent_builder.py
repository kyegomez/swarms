import os
from typing import Any, List, Optional, Tuple

from loguru import logger
from pydantic import BaseModel, Field

from swarms.structs.agent import Agent
from swarms.utils.function_caller_model import OpenAIFunctionCaller

BOSS_SYSTEM_PROMPT = """
# Swarm Intelligence Orchestrator

You are the Chief Orchestrator of a sophisticated agent swarm. Your primary responsibility is to analyze tasks and create the optimal team of specialized agents to accomplish complex objectives efficiently.

## Agent Creation Protocol

1. **Task Analysis**:
   - Thoroughly analyze the user's task to identify all required skills, knowledge domains, and subtasks
   - Break down complex problems into discrete components that can be assigned to specialized agents
   - Identify potential challenges and edge cases that might require specialized handling

2. **Agent Design Principles**:
   - Create highly specialized agents with clearly defined roles and responsibilities
   - Design each agent with deep expertise in their specific domain
   - Provide agents with comprehensive and extremely extensive system prompts that include:
     * Precise definition of their role and scope of responsibility
     * Detailed methodology for approaching problems in their domain
     * Specific techniques, frameworks, and mental models to apply
     * Guidelines for output format and quality standards
     * Instructions for collaboration with other agents
     * In-depth examples and scenarios to illustrate expected behavior and decision-making processes
     * Extensive background information relevant to the tasks they will undertake

3. **Cognitive Enhancement**:
   - Equip agents with advanced reasoning frameworks:
     * First principles thinking to break down complex problems
     * Systems thinking to understand interconnections
     * Lateral thinking for creative solutions
     * Critical thinking to evaluate information quality
   - Implement specialized thought patterns:
     * Step-by-step reasoning for complex problems
     * Hypothesis generation and testing
     * Counterfactual reasoning to explore alternatives
     * Analogical reasoning to apply solutions from similar domains

4. **Swarm Architecture**:
   - Design optimal agent interaction patterns based on task requirements
   - Consider hierarchical, networked, or hybrid structures
   - Establish clear communication protocols between agents
   - Define escalation paths for handling edge cases

5. **Agent Specialization Examples**:
   - Research Agents: Literature review, data gathering, information synthesis
   - Analysis Agents: Data processing, pattern recognition, insight generation
   - Creative Agents: Idea generation, content creation, design thinking
   - Planning Agents: Strategy development, resource allocation, timeline creation
   - Implementation Agents: Code writing, document drafting, execution planning
   - Quality Assurance Agents: Testing, validation, error detection
   - Integration Agents: Combining outputs, ensuring consistency, resolving conflicts

## Output Format

For each agent, provide:

1. **Agent Name**: Clear, descriptive title reflecting specialization
2. **Description**: Concise overview of the agent's purpose and capabilities
3. **System Prompt**: Comprehensive and extremely extensive instructions including:
   - Role definition and responsibilities
   - Specialized knowledge and methodologies
   - Thinking frameworks and problem-solving approaches
   - Output requirements and quality standards
   - Collaboration guidelines with other agents
   - Detailed examples and context to ensure clarity and effectiveness

## Optimization Guidelines

- Create only the agents necessary for the task - no more, no less
- Ensure each agent has a distinct, non-overlapping area of responsibility
- Design system prompts that maximize agent performance through clear guidance and specialized knowledge
- Balance specialization with the need for effective collaboration
- Prioritize agents that address the most critical aspects of the task

Remember: Your goal is to create a swarm of agents that collectively possesses the intelligence, knowledge, and capabilities to deliver exceptional results for the user's task.
"""


class AgentSpec(BaseModel):
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
        description="The name of the AI model that the agent will utilize for processing tasks and generating outputs. For example: gpt-4o, gpt-4o-mini, openai/o3-mini"
    )
    auto_generate_prompt: Optional[bool] = Field(
        description="A flag indicating whether the agent should automatically create prompts based on the task requirements."
    )
    max_tokens: Optional[int] = Field(
        None,
        description="The maximum number of tokens that the agent is allowed to generate in its responses, limiting output length.",
    )
    temperature: Optional[float] = Field(
        description="A parameter that controls the randomness of the agent's output; lower values result in more deterministic responses."
    )
    role: Optional[str] = Field(
        description="The designated role of the agent within the swarm, which influences its behavior and interaction with other agents."
    )
    max_loops: Optional[int] = Field(
        description="The maximum number of times the agent is allowed to repeat its task, enabling iterative processing if necessary."
    )


class Agents(BaseModel):
    """Configuration for a collection of agents that work together as a swarm to accomplish tasks."""

    agents: List[AgentSpec] = Field(
        description="A list containing the specifications of each agent that will participate in the swarm, detailing their roles and functionalities."
    )


class AgentsBuilder:
    """A class that automatically builds and manages swarms of AI agents.

    This class handles the creation, coordination and execution of multiple AI agents working
    together as a swarm to accomplish complex tasks. It uses a boss agent to delegate work
    and create new specialized agents as needed.

    Args:
        name (str): The name of the swarm
        description (str): A description of the swarm's purpose
        verbose (bool, optional): Whether to output detailed logs. Defaults to True.
        max_loops (int, optional): Maximum number of execution loops. Defaults to 1.
    """

    def __init__(
        self,
        name: str = "swarm-creator-01",
        description: str = "This is a swarm that creates swarms",
        verbose: bool = True,
        max_loops: int = 1,
        model_name: str = "gpt-4o",
        return_dictionary: bool = True,
        system_prompt: str = BOSS_SYSTEM_PROMPT,
    ):
        self.name = name
        self.description = description
        self.verbose = verbose
        self.max_loops = max_loops
        self.agents_pool = []
        self.model_name = model_name
        self.return_dictionary = return_dictionary
        self.system_prompt = system_prompt
        logger.info(
            f"Initialized AutoSwarmBuilder: {name} {description}"
        )

    def run(
        self, task: str, image_url: str = None, *args, **kwargs
    ) -> Tuple[List[Agent], int]:
        """Run the swarm on a given task.

        Args:
            task (str): The task to be accomplished
            image_url (str, optional): URL of an image input if needed. Defaults to None.
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments

        Returns:
            The output from the swarm's execution
        """
        logger.info(f"Running swarm on task: {task}")
        agents = self._create_agents(task, image_url, *args, **kwargs)

        return agents

    def _create_agents(self, task: str, *args, **kwargs):
        """Create the necessary agents for a task.

        Args:
            task (str): The task to create agents for
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments

        Returns:
            list: List of created agents
        """
        logger.info("Creating agents for task")
        model = OpenAIFunctionCaller(
            system_prompt=self.system_prompt,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1,
            base_model=Agents,
            model_name=self.model_name,
            max_tokens=8192,
        )

        agents_dictionary = model.run(task)
        print(agents_dictionary)

        print(type(agents_dictionary))
        logger.info("Agents successfully created")
        logger.info(f"Agents: {len(agents_dictionary.agents)}")

        if self.return_dictionary:
            logger.info("Returning dictionary")

            # Convert swarm config to dictionary
            agents_dictionary = agents_dictionary.model_dump()
            return agents_dictionary
        else:
            logger.info("Returning agents")
            return self.create_agents(agents_dictionary)

    def create_agents(self, agents_dictionary: Any):
        # Create agents from config
        agents = []
        for agent_config in agents_dictionary.agents:
            # Convert dict to AgentConfig if needed
            if isinstance(agent_config, dict):
                agent_config = Agents(**agent_config)

            agent = self.build_agent(
                agent_name=agent_config.model_name,
                agent_description=agent_config.description,
                agent_system_prompt=agent_config.system_prompt,
                model_name=agent_config.model_name,
                max_loops=agent_config.max_loops,
                dynamic_temperature_enabled=True,
                auto_generate_prompt=agent_config.auto_generate_prompt,
                role=agent_config.role,
                max_tokens=agent_config.max_tokens,
                temperature=agent_config.temperature,
            )
            agents.append(agent)

        return agents

    def build_agent(
        self,
        agent_name: str,
        agent_description: str,
        agent_system_prompt: str,
        max_loops: int = 1,
        model_name: str = "gpt-4o",
        dynamic_temperature_enabled: bool = True,
        auto_generate_prompt: bool = False,
        role: str = "worker",
        max_tokens: int = 8192,
        temperature: float = 0.5,
    ):
        """Build a single agent with the given specifications.

        Args:
            agent_name (str): Name of the agent
            agent_description (str): Description of the agent's purpose
            agent_system_prompt (str): The system prompt for the agent

        Returns:
            Agent: The constructed agent instance
        """
        logger.info(f"Building agent: {agent_name}")
        agent = Agent(
            agent_name=agent_name,
            description=agent_description,
            system_prompt=agent_system_prompt,
            model_name=model_name,
            max_loops=max_loops,
            dynamic_temperature_enabled=dynamic_temperature_enabled,
            context_length=200000,
            output_type="str",  # "json", "dict", "csv" OR "string" soon "yaml" and
            streaming_on=False,
            auto_generate_prompt=auto_generate_prompt,
            role=role,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return agent


# if __name__ == "__main__":
#     builder = AgentsBuilder(model_name="gpt-4o")
#     agents = builder.run("Create a swarm that can write a book about the history of the world")
#     print(agents)
#     print(type(agents))
