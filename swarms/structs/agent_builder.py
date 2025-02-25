import os
from typing import List

from pydantic import BaseModel, Field

from swarms.structs.agent import Agent
from swarms.utils.function_caller_model import OpenAIFunctionCaller
from swarms.utils.loguru_logger import initialize_logger
from swarms.structs.agents_available import showcase_available_agents
from swarms.structs.swarms_api import AgentInput as AgentConfig

logger = initialize_logger(log_folder="auto_swarm_builder")


class Agents(BaseModel):
    """Configuration for a list of agents"""

    agents: List[AgentConfig] = Field(
        description="The list of agents that make up the swarm",
    )


BOSS_SYSTEM_PROMPT = """
Manage a swarm of worker agents to efficiently serve the user by deciding whether to create new agents or delegate tasks. Ensure operations are efficient and effective.

### Instructions:

1. **Task Assignment**:
   - Analyze available worker agents when a task is presented.
   - Delegate tasks to existing agents with clear, direct, and actionable instructions if an appropriate agent is available.
   - If no suitable agent exists, create a new agent with a fitting system prompt to handle the task.

2. **Agent Creation**:
   - Name agents according to the task they are intended to perform (e.g., "Twitter Marketing Agent").
   - Provide each new agent with a concise and clear system prompt that includes its role, objectives, and any tools it can utilize.

3. **Efficiency**:
   - Minimize redundancy and maximize task completion speed.
   - Avoid unnecessary agent creation if an existing agent can fulfill the task.

4. **Communication**:
   - Be explicit in task delegation instructions to avoid ambiguity and ensure effective task execution.
   - Require agents to report back on task completion or encountered issues.

5. **Reasoning and Decisions**:
   - Offer brief reasoning when selecting or creating agents to maintain transparency.
   - Avoid using an agent if unnecessary, with a clear explanation if no agents are suitable for a task.

# Output Format

Present your plan in clear, bullet-point format or short concise paragraphs, outlining task assignment, agent creation, efficiency strategies, and communication protocols.

# Notes

- Preserve transparency by always providing reasoning for task-agent assignments and creation.
- Ensure instructions to agents are unambiguous to minimize error.

"""


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
        name: str = None,
        description: str = None,
        verbose: bool = True,
        max_loops: int = 1,
    ):
        self.name = name
        self.description = description
        self.verbose = verbose
        self.max_loops = max_loops
        self.agents_pool = []

        logger.info(
            f"Initialized AutoSwarmBuilder: {name} {description}"
        )

    def run(self, task: str, image_url: str = None, *args, **kwargs):
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
            system_prompt=BOSS_SYSTEM_PROMPT,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1,
            base_model=Agents,
        )

        agents_dictionary = model.run(task)
        logger.info(f"Agents dictionary: {agents_dictionary}")

        # Convert dictionary to SwarmConfig if needed
        if isinstance(agents_dictionary, dict):
            agents_dictionary = Agents(**agents_dictionary)

        # Create agents from config
        agents = []
        for agent_config in agents_dictionary.agents:
            # Convert dict to AgentConfig if needed
            if isinstance(agent_config, dict):
                agent_config = AgentConfig(**agent_config)

            agent = self.build_agent(
                agent_name=agent_config.name,
                agent_description=agent_config.description,
                agent_system_prompt=agent_config.system_prompt,
                model_name=agent_config.model_name,
                max_loops=agent_config.max_loops,
                dynamic_temperature_enabled=agent_config.dynamic_temperature_enabled,
                auto_generate_prompt=agent_config.auto_generate_prompt,
                role=agent_config.role,
                max_tokens=agent_config.max_tokens,
                temperature=agent_config.temperature,
            )
            agents.append(agent)

        # Showcasing available agents
        agents_available = showcase_available_agents(
            name=self.name,
            description=self.description,
            agents=agents,
        )

        for agent in agents:
            agent.system_prompt += "\n" + agents_available

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
