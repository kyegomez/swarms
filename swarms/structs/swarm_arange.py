import threading
import uuid
from typing import Any, Callable, Dict, List, Optional

from swarms.utils.any_to_str import any_to_str
from swarms.utils.loguru_logger import initialize_logger
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    HistoryOutputType,
)
from swarms.security import SwarmShieldIntegration, ShieldConfig

logger = initialize_logger(log_folder="swarm_arange")


def swarm_id():
    return uuid.uuid4().hex


class SwarmRearrange:
    """
    A class representing a swarm of swarms for rearranging tasks.

    Attributes:
        id (str): Unique identifier for the swarm arrangement
        name (str): Name of the swarm arrangement
        description (str): Description of what this swarm arrangement does
        swarms (dict): A dictionary of swarms, where the key is the swarm's name and the value is the swarm object
        flow (str): The flow pattern of the tasks
        max_loops (int): The maximum number of loops to run the swarm
        verbose (bool): A flag indicating whether to log verbose messages
        human_in_the_loop (bool): A flag indicating whether human intervention is required
        custom_human_in_the_loop (Callable[[str], str], optional): A custom function for human-in-the-loop intervention
        return_json (bool): A flag indicating whether to return the result in JSON format
        swarm_history (dict): A dictionary to keep track of the history of each swarm
        lock (threading.Lock): A lock for thread-safe operations
    """

    def __init__(
        self,
        id: str = swarm_id(),
        name: str = "SwarmRearrange",
        description: str = "A swarm of swarms for rearranging tasks.",
        swarms: List[Any] = [],
        flow: str = None,
        max_loops: int = 1,
        verbose: bool = True,
        human_in_the_loop: bool = False,
        custom_human_in_the_loop: Optional[
            Callable[[str], str]
        ] = None,
        return_json: bool = False,
        output_type: HistoryOutputType = "dict-all-except-first",
        shield_config: Optional[ShieldConfig] = None,
        enable_security: bool = True,
        security_level: str = "standard",
        *args,
        **kwargs,
    ):
        """
        Initializes the SwarmRearrange object.

        Args:
            id (str): Unique identifier for the swarm arrangement. Defaults to generated UUID.
            name (str): Name of the swarm arrangement. Defaults to "SwarmRearrange".
            description (str): Description of what this swarm arrangement does.
            swarms (List[swarm]): A list of swarm objects. Defaults to empty list.
            flow (str): The flow pattern of the tasks. Defaults to None.
            max_loops (int): Maximum number of loops to run. Defaults to 1.
            verbose (bool): Whether to log verbose messages. Defaults to True.
            human_in_the_loop (bool): Whether human intervention is required. Defaults to False.
            custom_human_in_the_loop (Callable): Custom function for human intervention. Defaults to None.
            return_json (bool): Whether to return results as JSON. Defaults to False.
            output_type (HistoryOutputType): Type of output format. Defaults to "dict-all-except-first".
            shield_config (ShieldConfig, optional): Security configuration for SwarmShield integration. Defaults to None.
            enable_security (bool, optional): Whether to enable SwarmShield security features. Defaults to True.
            security_level (str, optional): Pre-defined security level. Options: "basic", "standard", "enhanced", "maximum". Defaults to "standard".
        """
        self.id = id
        self.name = name
        self.description = description
        self.swarms = {swarm.name: swarm for swarm in swarms}
        self.flow = flow if flow is not None else ""
        self.max_loops = max_loops if max_loops > 0 else 1
        self.verbose = verbose
        self.human_in_the_loop = human_in_the_loop
        self.custom_human_in_the_loop = custom_human_in_the_loop
        self.output_type = output_type
        self.return_json = return_json

        # Initialize SwarmShield integration
        self._initialize_swarm_shield(shield_config, enable_security, security_level)

        self.swarm_history = {swarm.name: [] for swarm in swarms}
        self.lock = threading.Lock()
        self.id = uuid.uuid4().hex if id is None else id

        # Run the reliability checks
        self.reliability_checks()

        # Conversation
        self.conversation = Conversation()

    def reliability_checks(self):
        logger.info("Running reliability checks.")
        if not self.swarms:
            raise ValueError("No swarms found in the swarm.")

        if not self.flow:
            raise ValueError("No flow found in the swarm.")

        if self.max_loops <= 0:
            raise ValueError("Max loops must be a positive integer.")

        logger.info(
            "SwarmRearrange initialized with swarms: {}".format(
                list(self.swarms.keys())
            )
        )

    def set_custom_flow(self, flow: str):
        self.flow = flow
        logger.info(f"Custom flow set: {flow}")

    def add_swarm(self, swarm: Any):
        """
        Adds an swarm to the swarm.

        Args:
            swarm (swarm): The swarm to be added.
        """
        logger.info(f"Adding swarm {swarm.name} to the swarm.")
        self.swarms[swarm.name] = swarm

    def track_history(
        self,
        swarm_name: str,
        result: str,
    ):
        self.swarm_history[swarm_name].append(result)

    def remove_swarm(self, swarm_name: str):
        """
        Removes an swarm from the swarm.

        Args:
            swarm_name (str): The name of the swarm to be removed.
        """
        del self.swarms[swarm_name]

    def add_swarms(self, swarms: List[Any]):
        """
        Adds multiple swarms to the swarm.

        Args:
            swarms (List[swarm]): A list of swarm objects.
        """
        for swarm in swarms:
            self.swarms[swarm.name] = swarm

    def validate_flow(self):
        """
        Validates the flow pattern.

        Raises:
            ValueError: If the flow pattern is incorrectly formatted or contains duplicate swarm names.

        Returns:
            bool: True if the flow pattern is valid.
        """
        if "->" not in self.flow:
            raise ValueError(
                "Flow must include '->' to denote the direction of the task."
            )

        swarms_in_flow = []

        # Arrow
        tasks = self.flow.split("->")

        # For the task in tasks
        for task in tasks:
            swarm_names = [name.strip() for name in task.split(",")]

            # Loop over the swarm names
            for swarm_name in swarm_names:
                if (
                    swarm_name not in self.swarms
                    and swarm_name != "H"
                ):
                    raise ValueError(
                        f"swarm '{swarm_name}' is not registered."
                    )
                swarms_in_flow.append(swarm_name)

        # If the length of the swarms does not equal the length of the swarms in flow
        if len(set(swarms_in_flow)) != len(swarms_in_flow):
            raise ValueError(
                "Duplicate swarm names in the flow are not allowed."
            )

        logger.info("Flow is valid.")
        return True

    def run(
        self,
        task: str = None,
        img: str = None,
        custom_tasks: Optional[Dict[str, str]] = None,
        *args,
        **kwargs,
    ):
        """
        Runs the swarm to rearrange the tasks.

        Args:
            task: The initial task to be processed.
            img: An optional image input.
            custom_tasks: A dictionary of custom tasks for specific swarms.

        Returns:
            str: The final processed task.
        """
        try:
            if not self.validate_flow():
                return "Invalid flow configuration."

            tasks = self.flow.split("->")
            current_task = task

            # Check if custom_tasks is a dictionary and not empty
            if isinstance(custom_tasks, dict) and custom_tasks:
                c_swarm_name, c_task = next(
                    iter(custom_tasks.items())
                )

                # Find the position of the custom swarm in the tasks list
                if c_swarm_name in tasks:
                    position = tasks.index(c_swarm_name)

                    # If there is a previous swarm, merge its task with the custom tasks
                    if position > 0:
                        tasks[position - 1] += "->" + c_task
                    else:
                        # If there is no previous swarm, just insert the custom tasks
                        tasks.insert(position, c_task)

            # Set the loop counter
            loop_count = 0
            while loop_count < self.max_loops:
                for task in tasks:
                    swarm_names = [
                        name.strip() for name in task.split(",")
                    ]
                    if len(swarm_names) > 1:
                        # Parallel processing
                        logger.info(
                            f"Running swarms in parallel: {swarm_names}"
                        )
                        results = []
                        for swarm_name in swarm_names:
                            if swarm_name == "H":
                                # Human in the loop intervention
                                if (
                                    self.human_in_the_loop
                                    and self.custom_human_in_the_loop
                                ):
                                    current_task = (
                                        self.custom_human_in_the_loop(
                                            current_task
                                        )
                                    )
                                else:
                                    current_task = input(
                                        "Enter your response: "
                                    )
                            else:
                                swarm = self.swarms[swarm_name]
                                result = swarm.run(
                                    current_task, img, *args, **kwargs
                                )
                                result = any_to_str(result)
                                self.conversation.add(
                                    role=swarm.name, content=result
                                )

                                logger.info(
                                    f"Swarm {swarm_name} returned result of type: {type(result)}"
                                )
                                if isinstance(result, bool):
                                    logger.warning(
                                        f"Swarm {swarm_name} returned a boolean value: {result}"
                                    )
                                    result = str(
                                        result
                                    )  # Convert boolean to string
                                results.append(result)

                        current_task = "; ".join(
                            str(r) for r in results if r is not None
                        )
                    else:
                        # Sequential processing
                        logger.info(
                            f"Running swarms sequentially: {swarm_names}"
                        )
                        swarm_name = swarm_names[0]
                        if swarm_name == "H":
                            # Human-in-the-loop intervention
                            if (
                                self.human_in_the_loop
                                and self.custom_human_in_the_loop
                            ):
                                current_task = (
                                    self.custom_human_in_the_loop(
                                        current_task
                                    )
                                )
                            else:
                                current_task = input(
                                    "Enter the next task: "
                                )
                        else:
                            swarm = self.swarms[swarm_name]
                            result = swarm.run(
                                current_task, img, *args, **kwargs
                            )
                            result = any_to_str(result)

                            self.conversation.add(
                                role=swarm.name, content=result
                            )
                            logger.info(
                                f"Swarm {swarm_name} returned result of type: {type(result)}"
                            )
                            if isinstance(result, bool):
                                logger.warning(
                                    f"Swarm {swarm_name} returned a boolean value: {result}"
                                )
                                result = str(
                                    result
                                )  # Convert boolean to string
                            current_task = (
                                result
                                if result is not None
                                else current_task
                            )
                loop_count += 1

            return current_task

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return str(e)

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
        else:
            self.swarm_shield = None

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

    def enable_security(self) -> None:
        """Enable SwarmShield security features."""
        if not self.swarm_shield:
            self._initialize_swarm_shield(enable_security=True, security_level=self.security_level)

    def disable_security(self) -> None:
        """Disable SwarmShield security features."""
        self.swarm_shield = None
        self.enable_security = False

    def cleanup_security(self) -> None:
        """Clean up SwarmShield resources."""
        if self.swarm_shield:
            self.swarm_shield.cleanup()


def swarm_arrange(
    name: str = "SwarmArrange-01",
    description: str = "Combine multiple swarms and execute them sequentially",
    swarms: List[Callable] = None,
    output_type: str = "json",
    flow: str = None,
    task: str = None,
    *args,
    **kwargs,
):
    """
    Orchestrates the execution of multiple swarms in a sequential manner.

    Args:
        name (str, optional): The name of the swarm arrangement. Defaults to "SwarmArrange-01".
        description (str, optional): A description of the swarm arrangement. Defaults to "Combine multiple swarms and execute them sequentially".
        swarms (List[Callable], optional): A list of swarm objects to be executed. Defaults to None.
        output_type (str, optional): The format of the output. Defaults to "json".
        flow (str, optional): The flow pattern of the tasks. Defaults to None.
        task (str, optional): The task to be executed by the swarms. Defaults to None.
        *args: Additional positional arguments to be passed to the SwarmRearrange object.
        **kwargs: Additional keyword arguments to be passed to the SwarmRearrange object.

    Returns:
        Any: The result of the swarm arrangement execution.
    """
    try:
        swarm_arrangement = SwarmRearrange(
            name,
            description,
            swarms,
            output_type,
            flow,
        )
        result = swarm_arrangement.run(task, *args, **kwargs)
        result = any_to_str(result)
        logger.info(
            f"Swarm arrangement {name} executed successfully with output type {output_type}."
        )
        return result
    except Exception as e:
        logger.error(
            f"An error occurred during swarm arrangement execution: {e}"
        )
        return str(e)
