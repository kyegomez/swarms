from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional, Union, Tuple
from datetime import datetime

from swarms.structs.agent import Agent
from swarms.structs.rearrange import AgentRearrange
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.output_types import OutputType
from swarms.security import SwarmShieldIntegration, ShieldConfig

logger = initialize_logger(log_folder="sequential_workflow")


class SequentialWorkflow:
    """
    A class that orchestrates the execution of a sequence of agents in a defined workflow.

    Args:
        name (str, optional): The name of the workflow. Defaults to "SequentialWorkflow".
        description (str, optional): A description of the workflow. Defaults to "Sequential Workflow, where agents are executed in a sequence."
        agents (List[Agent], optional): A list of agents that will be part of the workflow. Defaults to an empty list.
        max_loops (int, optional): The maximum number of times to execute the workflow. Defaults to 1.
        output_type (OutputType, optional): The format of the output from the workflow. Defaults to "dict".
        shared_memory_system (callable, optional): A callable for managing shared memory between agents. Defaults to None.
        shield_config (ShieldConfig, optional): Security configuration for SwarmShield integration. Defaults to None.
        enable_security (bool, optional): Whether to enable SwarmShield security features. Defaults to True.
        security_level (str, optional): Pre-defined security level. Options: "basic", "standard", "enhanced", "maximum". Defaults to "standard".
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Raises:
        ValueError: If the agents list is None or empty, or if max_loops is set to 0.
    """

    def __init__(
        self,
        id: str = "sequential_workflow",
        name: str = "SequentialWorkflow",
        description: str = "Sequential Workflow, where agents are executed in a sequence.",
        agents: List[Union[Agent, Callable]] = [],
        max_loops: int = 1,
        output_type: OutputType = "dict",
        shared_memory_system: callable = None,
        shield_config: Optional[ShieldConfig] = None,
        enable_security: bool = True,
        security_level: str = "standard",
        *args,
        **kwargs,
    ):
        self.id = id
        self.name = name
        self.description = description
        self.agents = agents
        self.max_loops = max_loops
        self.output_type = output_type
        self.shared_memory_system = shared_memory_system

        # Initialize SwarmShield integration
        self._initialize_swarm_shield(shield_config, enable_security, security_level)

        self.reliability_check()
        self.flow = self.sequential_flow()

        self.agent_rearrange = AgentRearrange(
            name=self.name,
            description=self.description,
            agents=self.agents,
            flow=self.flow,
            max_loops=self.max_loops,
            output_type=self.output_type,
        )

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
    def validate_task_with_shield(self, task: str, agent_name: str = "default") -> Tuple[bool, str, Optional[str]]:
        """Validate and sanitize task input using SwarmShield."""
        if self.swarm_shield:
            return self.swarm_shield.validate_task(task, agent_name)
        return True, task, None

    def validate_agent_config_with_shield(self, agent_config: dict, agent_name: str = "default") -> Tuple[bool, dict, Optional[str]]:
        """Validate agent configuration using SwarmShield."""
        if self.swarm_shield:
            return self.swarm_shield.validate_agent_config(agent_config, agent_name)
        return True, agent_config, None

    def process_agent_communication_with_shield(self, agent_name: str, message: str, direction: str = "outbound") -> Tuple[bool, str, Optional[str]]:
        """Process agent communication through SwarmShield security."""
        if self.swarm_shield:
            return self.swarm_shield.process_agent_communication(agent_name, message, direction)
        return True, message, None

    def check_rate_limit_with_shield(self, agent_name: str, request_size: int = 1) -> Tuple[bool, Optional[str]]:
        """Check rate limits for an agent using SwarmShield."""
        if self.swarm_shield:
            return self.swarm_shield.check_rate_limit(agent_name, request_size)
        return True, None

    def add_secure_message(self, conversation_id: str, agent_name: str, message: str) -> bool:
        """Add a message to secure conversation history."""
        if self.swarm_shield:
            return self.swarm_shield.add_secure_message(conversation_id, agent_name, message)
        return False

    def get_secure_messages(self, conversation_id: str) -> List[Tuple[str, str, datetime]]:
        """Get secure conversation messages."""
        if self.swarm_shield:
            return self.swarm_shield.get_secure_messages(conversation_id)
        return []

    def create_secure_conversation(self, name: str = "") -> Optional[str]:
        """Create a secure conversation."""
        if self.swarm_shield:
            return self.swarm_shield.create_secure_conversation(name)
        return None

    def filter_and_protect_output(self, output_data: Union[str, dict, List], agent_name: str, output_type: str = "text") -> Tuple[bool, Union[str, dict, List], Optional[str]]:
        """Filter and protect output data using SwarmShield."""
        if self.swarm_shield:
            return self.swarm_shield.filter_and_protect_output(output_data, agent_name, output_type)
        return True, output_data, None

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

    def sequential_flow(self):
        # Only create flow if agents exist
        if self.agents:
            # Create flow by joining agent names with arrows
            agent_names = []
            for agent in self.agents:
                try:
                    # Try to get agent_name, fallback to name if not available
                    agent_name = (
                        getattr(agent, "agent_name", None)
                        or agent.name
                    )
                    agent_names.append(agent_name)
                except AttributeError:
                    logger.warning(
                        f"Could not get name for agent {agent}"
                    )
                    continue

            if agent_names:
                flow = " -> ".join(agent_names)
            else:
                flow = ""
                logger.warning(
                    "No valid agent names found to create flow"
                )
        else:
            flow = ""
            logger.warning("No agents provided to create flow")

        return flow

    def reliability_check(self):
        if self.agents is None or len(self.agents) == 0:
            raise ValueError("Agents list cannot be None or empty")

        if self.max_loops == 0:
            raise ValueError("max_loops cannot be 0")

        logger.info("Checks completed; your swarm is ready.")

    def run(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        *args,
        **kwargs,
    ):
        """
        Executes a specified task through the agents in the dynamically constructed flow.

        Args:
            task (str): The task for the agents to execute.
            img (Optional[str]): An optional image input for the agents.
            device (str): The device to use for the agents to execute. Defaults to "cpu".
            all_cores (bool): Whether to utilize all CPU cores. Defaults to False.
            all_gpus (bool): Whether to utilize all available GPUs. Defaults to False.
            device_id (int): The specific device ID to use for execution. Defaults to 0.
            no_use_clusterops (bool): Whether to avoid using cluster operations. Defaults to True.

        Returns:
            str: The final result after processing through all agents.

        Raises:
            ValueError: If the task is None or empty.
            Exception: If any error occurs during task execution.
        """

        try:
            return self.agent_rearrange.run(
                task=task,
                img=img,
                # imgs=imgs,
                # *args,
                # **kwargs,
            )

        except Exception as e:
            logger.error(
                f"An error occurred while executing the task: {e}"
            )
            raise e

    def __call__(self, task: str, *args, **kwargs):
        return self.run(task, *args, **kwargs)

    def run_batched(self, tasks: List[str]) -> List[str]:
        """
        Executes a batch of tasks through the agents in the dynamically constructed flow.

        Args:
            tasks (List[str]): A list of tasks for the agents to execute.

        Returns:
            List[str]: A list of final results after processing through all agents.

        Raises:
            ValueError: If tasks is None or empty.
            Exception: If any error occurs during task execution.
        """
        if not tasks or not all(
            isinstance(task, str) for task in tasks
        ):
            raise ValueError(
                "Tasks must be a non-empty list of strings"
            )

        try:
            return [self.agent_rearrange.run(task) for task in tasks]
        except Exception as e:
            logger.error(
                f"An error occurred while executing the batch of tasks: {e}"
            )
            raise

    async def run_async(self, task: str) -> str:
        """
        Executes the specified task through the agents in the dynamically constructed flow asynchronously.

        Args:
            task (str): The task for the agents to execute.

        Returns:
            str: The final result after processing through all agents.

        Raises:
            ValueError: If task is None or empty.
            Exception: If any error occurs during task execution.
        """
        if not task or not isinstance(task, str):
            raise ValueError("Task must be a non-empty string")

        try:
            return await self.agent_rearrange.run_async(task)
        except Exception as e:
            logger.error(
                f"An error occurred while executing the task asynchronously: {e}"
            )
            raise

    async def run_concurrent(self, tasks: List[str]) -> List[str]:
        """
        Executes a batch of tasks through the agents in the dynamically constructed flow concurrently.

        Args:
            tasks (List[str]): A list of tasks for the agents to execute.

        Returns:
            List[str]: A list of final results after processing through all agents.

        Raises:
            ValueError: If tasks is None or empty.
            Exception: If any error occurs during task execution.
        """
        if not tasks or not all(
            isinstance(task, str) for task in tasks
        ):
            raise ValueError(
                "Tasks must be a non-empty list of strings"
            )

        try:
            with ThreadPoolExecutor() as executor:
                results = [
                    executor.submit(self.agent_rearrange.run, task)
                    for task in tasks
                ]
                return [
                    result.result()
                    for result in as_completed(results)
                ]
        except Exception as e:
            logger.error(
                f"An error occurred while executing the batch of tasks concurrently: {e}"
            )
            raise
