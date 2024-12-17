import os
from typing import List, Optional
from datetime import datetime

from pydantic import BaseModel, Field
from pydantic.v1 import validator
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

from swarm_models import OpenAIFunctionCaller, OpenAIChat
from swarms.structs.agent import Agent
from swarms.structs.swarm_router import SwarmRouter
from swarms.structs.agents_available import showcase_available_agents


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


class AgentConfig(BaseModel):
    """Configuration for an individual agent in a swarm"""

    name: str = Field(
        description="The name of the agent", example="Research-Agent"
    )
    description: str = Field(
        description="A description of the agent's purpose and capabilities",
        example="Agent responsible for researching and gathering information",
    )
    system_prompt: str = Field(
        description="The system prompt that defines the agent's behavior",
        example="You are a research agent. Your role is to gather and analyze information...",
    )

    @validator("name")
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError("Agent name cannot be empty")
        return v.strip()

    @validator("system_prompt")
    def validate_system_prompt(cls, v):
        if not v.strip():
            raise ValueError("System prompt cannot be empty")
        return v.strip()


class SwarmConfig(BaseModel):
    """Configuration for a swarm of cooperative agents"""

    name: str = Field(
        description="The name of the swarm",
        example="Research-Writing-Swarm",
    )
    description: str = Field(
        description="The description of the swarm's purpose and capabilities",
        example="A swarm of agents that work together to research topics and write articles",
    )
    agents: List[AgentConfig] = Field(
        description="The list of agents that make up the swarm",
        min_items=1,
    )

    @validator("agents")
    def validate_agents(cls, v):
        if not v:
            raise ValueError("Swarm must have at least one agent")
        return v


class AutoSwarmBuilder:
    """A class that automatically builds and manages swarms of AI agents with enhanced error handling."""

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        verbose: bool = True,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4",
    ):
        self.name = name or "DefaultSwarm"
        self.description = description or "Generic AI Agent Swarm"
        self.verbose = verbose
        self.agents_pool = []
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name

        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided either through initialization or environment variable"
            )

        logger.info(
            "Initialized AutoSwarmBuilder",
            extra={
                "swarm_name": self.name,
                "description": self.description,
                "model": self.model_name,
            },
        )

        # Initialize OpenAI chat model
        try:
            self.chat_model = OpenAIChat(
                openai_api_key=self.api_key,
                model_name=self.model_name,
                temperature=0.1,
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize OpenAI chat model: {str(e)}"
            )
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def run(self, task: str, image_url: Optional[str] = None) -> str:
        """Run the swarm on a given task with error handling and retries."""
        if not task or not task.strip():
            raise ValueError("Task cannot be empty")

        logger.info("Starting swarm execution", extra={"task": task})

        try:
            # Create agents for the task
            agents = self._create_agents(task, image_url)
            if not agents:
                raise ValueError(
                    "No agents were created for the task"
                )

            # Execute the task through the swarm router
            logger.info(
                "Routing task through swarm",
                extra={"num_agents": len(agents)},
            )
            output = self.swarm_router(agents, task, image_url)

            logger.info("Swarm execution completed successfully")
            return output

        except Exception as e:
            logger.error(
                f"Error during swarm execution: {str(e)}",
                exc_info=True,
            )
            raise

    def _create_agents(
        self, task: str, image_url: Optional[str] = None
    ) -> List[Agent]:
        """Create the necessary agents for a task with enhanced error handling."""
        logger.info("Creating agents for task", extra={"task": task})

        try:
            model = OpenAIFunctionCaller(
                system_prompt=BOSS_SYSTEM_PROMPT,
                api_key=self.api_key,
                temperature=0.1,
                base_model=SwarmConfig,
            )

            agents_config = model.run(task)
            print(f"{agents_config}")

            if isinstance(agents_config, dict):
                agents_config = SwarmConfig(**agents_config)

            # Update swarm configuration
            self.name = agents_config.name
            self.description = agents_config.description

            # Create agents from configuration
            agents = []
            for agent_config in agents_config.agents:
                if isinstance(agent_config, dict):
                    agent_config = AgentConfig(**agent_config)

                agent = self.build_agent(
                    agent_name=agent_config.name,
                    agent_description=agent_config.description,
                    agent_system_prompt=agent_config.system_prompt,
                )
                agents.append(agent)

            # Add available agents showcase to system prompts
            agents_available = showcase_available_agents(
                name=self.name,
                description=self.description,
                agents=agents,
            )

            for agent in agents:
                agent.system_prompt += "\n" + agents_available

            logger.info(
                "Successfully created agents",
                extra={"num_agents": len(agents)},
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
        """Build a single agent with enhanced error handling."""
        logger.info(
            "Building agent", extra={"agent_name": agent_name}
        )

        try:
            agent = Agent(
                agent_name=agent_name,
                description=agent_description,
                system_prompt=agent_system_prompt,
                llm=self.chat_model,
                autosave=True,
                dashboard=False,
                verbose=self.verbose,
                dynamic_temperature_enabled=True,
                saved_state_path=f"states/{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                user_name="swarms_corp",
                retry_attempts=3,
                context_length=200000,
                return_step_meta=False,
                output_type="str",
                streaming_on=False,
                auto_generate_prompt=True,
            )
            return agent

        except Exception as e:
            logger.error(
                f"Error building agent: {str(e)}", exc_info=True
            )
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def swarm_router(
        self,
        agents: List[Agent],
        task: str,
        image_url: Optional[str] = None,
    ) -> str:
        """Route tasks between agents in the swarm with error handling and retries."""
        logger.info(
            "Initializing swarm router",
            extra={"num_agents": len(agents)},
        )

        try:
            swarm_router_instance = SwarmRouter(
                name=self.name,
                description=self.description,
                agents=agents,
                swarm_type="auto",
            )

            formatted_task = f"{self.name} {self.description} {task}"
            result = swarm_router_instance.run(formatted_task)

            logger.info("Successfully completed swarm routing")
            return result

        except Exception as e:
            logger.error(
                f"Error in swarm router: {str(e)}", exc_info=True
            )
            raise


swarm = AutoSwarmBuilder(
    name="ChipDesign-Swarm",
    description="A swarm of specialized AI agents for chip design",
    api_key="your-api-key",  # Optional if set in environment
    model_name="gpt-4",  # Optional, defaults to gpt-4
)

result = swarm.run(
    "Design a new AI accelerator chip optimized for transformer model inference..."
)
