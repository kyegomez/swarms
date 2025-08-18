from pydantic.v1 import BaseModel
from typing import List, Callable, Optional
from swarms.utils.loguru_logger import initialize_logger
from swarms.security import SwarmShieldIntegration, ShieldConfig

logger = initialize_logger(log_folder="swarm_registry")


class SwarmRegistry(BaseModel):
    def __init__(
        self,
        swarm_pool: List[Callable] = [],
        shield_config: Optional[ShieldConfig] = None,
        enable_security: bool = True,
        security_level: str = "standard",
        **kwargs
    ):
        super().__init__(swarm_pool=swarm_pool, **kwargs)
        
        # Initialize SwarmShield integration
        self._initialize_swarm_shield(shield_config, enable_security, security_level)

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

    swarm_pool: List[Callable] = []

    def add(self, swarm: Callable, *args, **kwargs):
        """
        Adds a swarm to the registry.

        Args:
            swarm (Callable): The swarm to add to the registry.
        """
        self.swarm_pool.append(swarm, *args, **kwargs)

    def query(self, swarm_name: str) -> Callable:
        """
        Queries the registry for a swarm by name.

        Args:
            swarm_name (str): The name of the swarm to query.

        Returns:
            Callable: The swarm function corresponding to the given name.
        """
        if not self.swarm_pool:
            raise ValueError("No swarms found in registry")

        if not swarm_name:
            raise ValueError("No swarm name provided.")

        for swarm in self.swarm_pool:
            if swarm.__name__ == swarm_name:
                name = swarm.__name__
                description = (
                    swarm.__doc__.strip().split("\n")[0]
                    or swarm.description
                )
                agent_count = len(swarm.agents)
                task_count = len(swarm.tasks)

                log = f"Swarm: {name}\nDescription: {description}\nAgents: {agent_count}\nTasks: {task_count}"
                logger.info(log)

            return swarm

        raise ValueError(
            f"Swarm '{swarm_name}' not found in registry."
        )

    def remove(self, swarm_name: str):
        """
        Removes a swarm from the registry by name.

        Args:
            swarm_name (str): The name of the swarm to remove.
        """
        for swarm in self.swarm_pool:
            if swarm.__name__ == swarm_name:
                self.swarm_pool.remove(swarm)
                return
        raise ValueError(
            f"Swarm '{swarm_name}' not found in registry."
        )

    def list_swarms(self) -> List[str]:
        """
        Lists the names of all swarms in the registry.

        Returns:
            List[str]: A list of swarm names.
        """
        if not self.swarm_pool:
            raise ValueError("No swarms found in registry.")

        for swarm in self.swarm_pool:
            name = swarm.__name__
            description = (
                swarm.__doc__.strip().split("\n")[0]
                or swarm.description
            )
            agent_count = len(swarm.agents)
            task_count = len(swarm.tasks)

            log = f"Swarm: {name}\nDescription: {description}\nAgents: {agent_count}\nTasks: {task_count}"
            logger.info(log)

        return [swarm.__name__ for swarm in self.swarm_pool]

    def run(self, swarm_name: str, *args, **kwargs):
        """
        Runs a swarm by name with the given arguments.

        Args:
            swarm_name (str): The name of the swarm to run.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The result of running the swarm.
        """
        swarm = self.query(swarm_name)
        return swarm(*args, **kwargs)

    def add_list_of_swarms(self, swarms: List[Callable]):
        """
        Adds a list of swarms to the registry.

        Args:
            swarms (List[Callable]): A list of swarms to add to the registry.
        """
        for swarm in swarms:
            self.add(swarm)

        return self.swarm_pool

    def query_multiple_of_swarms(
        self, swarm_names: List[str]
    ) -> List[Callable]:
        """
        Queries the registry for multiple swarms by name.

        Args:
            swarm_names (List[str]): A list of swarm names to query.

        Returns:
            List[Callable]: A list of swarm functions corresponding to the given names.
        """
        return [self.query(swarm_name) for swarm_name in swarm_names]

    def remove_list_of_swarms(self, swarm_names: List[str]):
        """
        Removes a list of swarms from the registry by name.

        Args:
            swarm_names (List[str]): A list of swarm names to remove.
        """
        for swarm_name in swarm_names:
            self.remove(swarm_name)

        return self.swarm_pool

    def run_multiple_of_swarms(
        self, swarm_names: List[str], *args, **kwargs
    ):
        """
        Runs a list of swarms by name with the given arguments.

        Args:
            swarm_names (List[str]): A list of swarm names to run.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List[Any]: A list of results of running the swarms.
        """
        return [
            self.run(swarm_name, *args, **kwargs)
            for swarm_name in swarm_names
        ]


# Decorator to add a function to the registry
def swarm_registry():
    """
    Decorator to add a function to the registry.

    Args:
        swarm_registry (SwarmRegistry): The swarm registry instance.

    Returns:
        Callable: The decorated function.
    """

    def decorator(func, *args, **kwargs):
        try:
            swarm_registry = SwarmRegistry()
            swarm_registry.add(func, *args, **kwargs)
            logger.info(
                f"Added swarm '{func.__name__}' to the registry."
            )
            return func
        except Exception as e:
            logger.error(str(e))
            raise

    return decorator
