import os
from typing import List, Optional
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.multi_agent_exec import get_swarms_info
from swarms.structs.swarm_router import SwarmRouter
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union, Callable
from swarms.utils.history_output_formatter import HistoryOutputType
from swarms.security import SwarmShieldIntegration, ShieldConfig

tools = [
    {
        "type": "function",
        "function": {
            "name": "select_swarm",
            "description": "Analyzes the input task and selects the most appropriate swarm configuration, outputting both the swarm name and the formatted task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "The reasoning behind the selection of the swarm and task description.",
                    },
                    "swarm_name": {
                        "type": "string",
                        "description": "The name of the selected swarm that is most appropriate for handling the given task.",
                    },
                    "task_description": {
                        "type": "string",
                        "description": "A clear and structured description of the task to be performed by the swarm.",
                    },
                },
                "required": [
                    "reasoning",
                    "swarm_name",
                    "task_description",
                ],
            },
        },
    },
]

router_system_prompt = """
You are an intelligent Router Agent responsible for analyzing tasks and directing them to the most appropriate swarm in our system. Your role is critical in ensuring optimal task execution and resource utilization.

Key Responsibilities:
1. Task Analysis:
   - Carefully analyze the input task's requirements, complexity, and domain
   - Identify key components and dependencies
   - Determine the specialized skills needed for completion

2. Swarm Selection Criteria:
   - Match task requirements with swarm capabilities
   - Consider swarm specializations and past performance
   - Evaluate computational resources needed
   - Account for task priority and time constraints

3. Decision Making Framework:
   - Use a systematic approach to evaluate all available swarms
   - Consider load balancing across the system
   - Factor in swarm availability and current workload
   - Assess potential risks and failure points

4. Output Requirements:
   - Provide clear justification for swarm selection
   - Structure the task description in a way that maximizes swarm efficiency
   - Include any relevant context or constraints
   - Ensure all critical information is preserved

Best Practices:
- Always prefer specialized swarms for domain-specific tasks
- Consider breaking complex tasks into subtasks when appropriate
- Maintain consistency in task formatting across different swarms
- Include error handling considerations in task descriptions

Your output must strictly follow the required format:
{
    "swarm_name": "Name of the selected swarm",
    "task_description": "Detailed and structured task description"
}

Remember: Your selection directly impacts the overall system performance and task completion success rate. Take all factors into account before making your final decision.
"""


class HybridHierarchicalClusterSwarm:
    """
    A class representing a Hybrid Hierarchical-Cluster Swarm that routes tasks to appropriate swarms.

    Attributes:
        name (str): The name of the swarm.
        description (str): A description of the swarm's functionality.
        swarms (List[SwarmRouter]): A list of available swarm routers.
        max_loops (int): The maximum number of loops for task processing.
        output_type (str): The format of the output (e.g., list).
        conversation (Conversation): An instance of the Conversation class to manage interactions.
        router_agent (Agent): An instance of the Agent class responsible for routing tasks.
    """

    def __init__(
        self,
        name: str = "Hybrid Hierarchical-Cluster Swarm",
        description: str = "A swarm that uses a hybrid hierarchical-peer model to solve complex tasks.",
        swarms: List[Union[SwarmRouter, Callable]] = [],
        max_loops: int = 1,
        output_type: HistoryOutputType = "list",
        router_agent_model_name: str = "gpt-4o-mini",
        shield_config: Optional[ShieldConfig] = None,
        enable_security: bool = True,
        security_level: str = "standard",
        *args,
        **kwargs,
    ):
        self.name = name
        self.description = description
        self.swarms = swarms
        self.max_loops = max_loops
        self.output_type = output_type

        # Initialize SwarmShield integration
        self._initialize_swarm_shield(shield_config, enable_security, security_level)

        self.conversation = Conversation()

        self.router_agent = Agent(
            agent_name="Router Agent",
            agent_description="A router agent that routes tasks to the appropriate swarms.",
            system_prompt=f"{router_system_prompt}\n\n{get_swarms_info(swarms=self.swarms)}",
            tools_list_dictionary=tools,
            model_name=router_agent_model_name,
            max_loops=1,
            output_type="final",
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

    def convert_str_to_dict(self, response: str):
        # Handle response whether it's a string or dictionary
        if isinstance(response, str):
            try:
                import json

                response = json.loads(response)
            except json.JSONDecodeError:
                raise ValueError(
                    "Invalid JSON response from router agent"
                )

        return response

    def run(self, task: str, *args, **kwargs):
        """
        Runs the routing process for a given task.

        Args:
            task (str): The task to be processed by the swarm.

        Returns:
            str: The formatted history output of the conversation.

        Raises:
            ValueError: If the task is empty or invalid.
        """
        if not task:
            raise ValueError("Task cannot be empty.")

        self.conversation.add(role="User", content=task)

        response = self.router_agent.run(task=task)

        if isinstance(response, str):
            response = self.convert_str_to_dict(response)
        else:
            pass

        swarm_name = response.get("swarm_name")
        task_description = response.get("task_description")

        if not swarm_name or not task_description:
            raise ValueError(
                "Invalid response from router agent: both 'swarm_name' and 'task_description' must be present. "
                f"Received: swarm_name={swarm_name}, task_description={task_description}. "
                f"Please check the response format from the model: {self.router_agent.model_name}."
            )

        self.route_task(swarm_name, task_description)

        return history_output_formatter(
            self.conversation, self.output_type
        )

    def find_swarm_by_name(self, swarm_name: str):
        """
        Finds a swarm by its name.

        Args:
            swarm_name (str): The name of the swarm to find.

        Returns:
            SwarmRouter: The found swarm router, or None if not found.
        """
        for swarm in self.swarms:
            if swarm.name == swarm_name:
                return swarm
        return None

    def route_task(self, swarm_name: str, task_description: str):
        """
        Routes the task to the specified swarm.

        Args:
            swarm_name (str): The name of the swarm to route the task to.
            task_description (str): The description of the task to be executed.

        Raises:
            ValueError: If the swarm is not found.
        """
        swarm = self.find_swarm_by_name(swarm_name)

        if swarm:
            output = swarm.run(task_description)
            self.conversation.add(role=swarm.name, content=output)
        else:
            raise ValueError(f"Swarm '{swarm_name}' not found.")

    def batched_run(self, tasks: List[str]):
        """
        Runs the routing process for a list of tasks in batches.

        Args:
            tasks (List[str]): A list of tasks to be processed by the swarm.

        Returns:
            List[str]: A list of formatted history outputs for each batch.

        Raises:
            ValueError: If the task list is empty or invalid.
        """
        if not tasks:
            raise ValueError("Task list cannot be empty.")

        max_workers = os.cpu_count() * 2

        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the executor
            future_to_task = {
                executor.submit(self.run, task): task
                for task in tasks
            }

            # Collect results as they complete
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # Handle any errors that occurred during task execution
                    results.append(f"Error processing task: {str(e)}")

        return results
