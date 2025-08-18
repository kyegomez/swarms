import os
from typing import List, Optional


from swarms.structs.agent import Agent
from swarms.prompts.ag_prompt import aggregator_system_prompt_main
from swarms.structs.ma_utils import list_all_agents
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.loguru_logger import initialize_logger
import concurrent.futures
from swarms.utils.output_types import OutputType
from swarms.structs.conversation import Conversation
from swarms.security import SwarmShieldIntegration, ShieldConfig


logger = initialize_logger(log_folder="mixture_of_agents")


class MixtureOfAgents:
    """
    A class to manage and run a mixture of agents, aggregating their responses.
    """

    def __init__(
        self,
        name: str = "MixtureOfAgents",
        description: str = "A class to run a mixture of agents and aggregate their responses.",
        agents: List[Agent] = None,
        aggregator_agent: Agent = None,
        aggregator_system_prompt: str = aggregator_system_prompt_main,
        layers: int = 3,
        max_loops: int = 1,
        output_type: OutputType = "final",
        aggregator_model_name: str = "claude-3-5-sonnet-20240620",
        shield_config: Optional[ShieldConfig] = None,
        enable_security: bool = True,
        security_level: str = "standard",
    ) -> None:
        """
        Initialize the Mixture of Agents class with agents and configuration.

        Args:
            name (str, optional): The name of the mixture of agents. Defaults to "MixtureOfAgents".
            description (str, optional): A description of the mixture of agents. Defaults to "A class to run a mixture of agents and aggregate their responses.".
            agents (List[Agent], optional): A list of reference agents to be used in the mixture. Defaults to [].
            aggregator_agent (Agent, optional): The aggregator agent to be used in the mixture. Defaults to None.
            aggregator_system_prompt (str, optional): The system prompt for the aggregator agent. Defaults to "".
            layers (int, optional): The number of layers to process in the mixture. Defaults to 3.
            shield_config (ShieldConfig, optional): Security configuration for SwarmShield integration. Defaults to None.
            enable_security (bool, optional): Whether to enable SwarmShield security features. Defaults to True.
            security_level (str, optional): Pre-defined security level. Options: "basic", "standard", "enhanced", "maximum". Defaults to "standard".
        """
        self.name = name
        self.description = description
        self.agents = agents
        self.aggregator_agent = aggregator_agent
        self.aggregator_system_prompt = aggregator_system_prompt
        self.layers = layers
        self.max_loops = max_loops
        self.output_type = output_type
        self.aggregator_model_name = aggregator_model_name
        self.aggregator_agent = self.aggregator_agent_setup()

        # Initialize SwarmShield integration
        self._initialize_swarm_shield(shield_config, enable_security, security_level)

        self.reliability_check()

        self.conversation = Conversation()

        list_all_agents(
            agents=self.agents,
            conversation=self.conversation,
            description=self.description,
            name=self.name,
            add_to_conversation=True,
        )

    def aggregator_agent_setup(self):
        return Agent(
            agent_name="Aggregator Agent",
            description="An agent that aggregates the responses of the other agents.",
            system_prompt=aggregator_system_prompt_main,
            model_name=self.aggregator_model_name,
            temperature=0.5,
            max_loops=1,
            output_type="str-all-except-first",
        )

    def reliability_check(self) -> None:
        """
        Performs a reliability check on the Mixture of Agents class.
        """
        logger.info(
            "Checking the reliability of the Mixture of Agents class."
        )

        if len(self.agents) == 0:
            raise ValueError("No agents provided.")

        if not self.aggregator_agent:
            raise ValueError("No aggregator agent provided.")

        if not self.aggregator_system_prompt:
            raise ValueError("No aggregator system prompt provided.")

        if not self.layers:
            raise ValueError("No layers provided.")

        logger.info("Reliability check passed.")
        logger.info("Mixture of Agents class is ready for use.")

    def save_to_markdown_file(self, file_path: str = "moa.md"):
        with open(file_path, "w") as f:
            f.write(self.conversation.get_str())

    def step(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
    ):
        # self.conversation.add(role="User", content=task)

        # Run agents concurrently
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:
            # Submit all agent tasks and store with their index
            future_to_agent = {
                executor.submit(
                    agent.run, task=task, img=img, imgs=imgs
                ): agent
                for agent in self.agents
            }

            # Collect results and add to conversation in completion order
            for future in concurrent.futures.as_completed(
                future_to_agent
            ):
                agent = future_to_agent[future]
                output = future.result()
                self.conversation.add(role=agent.name, content=output)

        return self.conversation.get_str()

    def _run(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
    ):

        self.conversation.add(role="User", content=task)

        for i in range(self.layers):
            out = self.step(
                task=self.conversation.get_str(), img=img, imgs=imgs
            )
            task = out

        out = self.aggregator_agent.run(
            task=self.conversation.get_str()
        )

        self.conversation.add(
            role=self.aggregator_agent.agent_name, content=out
        )

        out = history_output_formatter(
            conversation=self.conversation, type=self.output_type
        )

        return out

    def run(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
    ):
        try:
            return self._run(task=task, img=img, imgs=imgs)
        except Exception as e:
            logger.error(f"Error running Mixture of Agents: {e}")
            return f"Error: {e}"

    def run_batched(self, tasks: List[str]) -> List[str]:
        """
        Run the mixture of agents for a batch of tasks.

        Args:
            tasks (List[str]): A list of tasks for the mixture of agents.

        Returns:
            List[str]: A list of responses from the mixture of agents.
        """
        return [self.run(task) for task in tasks]

    def run_concurrently(self, tasks: List[str]) -> List[str]:
        """
        Run the mixture of agents for a batch of tasks concurrently.
        """
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:
            futures = [
                executor.submit(self.run, task) for task in tasks
            ]
            return [
                future.result()
                for future in concurrent.futures.as_completed(futures)
            ]

    def _initialize_swarm_shield(
        self, 
        shield_config: Optional[ShieldConfig] = None,
        enable_security: bool = True,
        security_level: str = "standard"
    ) -> None:
        """Initialize SwarmShield integration for security features."""
        self.enable_security = enable_security
        self.security_level = security_level
        
        if enable_security:
            if shield_config is None:
                shield_config = ShieldConfig.get_security_level(security_level)
            
            self.swarm_shield = SwarmShieldIntegration(shield_config)
            logger.info(f"SwarmShield initialized with {security_level} security level")
        else:
            self.swarm_shield = None
            logger.info("SwarmShield security disabled")

    # Security methods
    def validate_task_with_shield(self, task: str) -> str:
        """Validate and sanitize task input using SwarmShield."""
        if self.swarm_shield:
            return self.swarm_shield.validate_and_protect_input(task)
        return task

    def validate_agent_config_with_shield(self, agent_config: dict) -> dict:
        """Validate agent configuration using SwarmShield."""
        if self.swarm_shield:
            return self.swarm_shield.validate_and_protect_input(str(agent_config))
        return agent_config

    def process_agent_communication_with_shield(self, message: str, agent_name: str) -> str:
        """Process agent communication through SwarmShield security."""
        if self.swarm_shield:
            return self.swarm_shield.process_agent_communication(message, agent_name)
        return message

    def check_rate_limit_with_shield(self, agent_name: str) -> bool:
        """Check rate limits for an agent using SwarmShield."""
        if self.swarm_shield:
            return self.swarm_shield.check_rate_limit(agent_name)
        return True

    def add_secure_message(self, message: str, agent_name: str) -> None:
        """Add a message to secure conversation history."""
        if self.swarm_shield:
            self.swarm_shield.add_secure_message(message, agent_name)

    def get_secure_messages(self) -> List[dict]:
        """Get secure conversation messages."""
        if self.swarm_shield:
            return self.swarm_shield.get_secure_messages()
        return []

    def get_security_stats(self) -> dict:
        """Get security statistics and metrics."""
        if self.swarm_shield:
            return self.swarm_shield.get_security_stats()
        return {"security_enabled": False}

    def update_shield_config(self, new_config: ShieldConfig) -> None:
        """Update SwarmShield configuration."""
        if self.swarm_shield:
            self.swarm_shield.update_config(new_config)
            logger.info("SwarmShield configuration updated")

    def enable_security(self) -> None:
        """Enable SwarmShield security features."""
        if not self.swarm_shield:
            self._initialize_swarm_shield(enable_security=True, security_level=self.security_level)
            logger.info("SwarmShield security enabled")

    def disable_security(self) -> None:
        """Disable SwarmShield security features."""
        self.swarm_shield = None
        self.enable_security = False
        logger.info("SwarmShield security disabled")

    def cleanup_security(self) -> None:
        """Clean up SwarmShield resources."""
        if self.swarm_shield:
            self.swarm_shield.cleanup()
            logger.info("SwarmShield resources cleaned up")
