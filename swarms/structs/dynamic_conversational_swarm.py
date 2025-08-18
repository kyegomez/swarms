import json
import random
from swarms.structs.agent import Agent
from typing import List, Optional
from swarms.structs.conversation import Conversation
from swarms.structs.ma_blocks import find_agent_by_name
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.any_to_str import any_to_str
from swarms.security import SwarmShieldIntegration, ShieldConfig

tools = [
    {
        "type": "function",
        "function": {
            "name": "select_agent",
            "description": "Analyzes the input task and selects the most appropriate agent configuration, outputting both the agent name and the formatted response.",
            "parameters": {
                "type": "object",
                "properties": {
                    "respond_or_no_respond": {
                        "type": "boolean",
                        "description": "Whether the agent should respond to the response or not.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "The reasoning behind the selection of the agent and response.",
                    },
                    "agent_name": {
                        "type": "string",
                        "description": "The name of the selected agent that is most appropriate for handling the given task.",
                    },
                    "response": {
                        "type": "string",
                        "description": "A clear and structured description of the response for the next agent.",
                    },
                },
                "required": [
                    "reasoning",
                    "agent_name",
                    "response",
                    "respond_or_no_respond",
                ],
            },
        },
    },
]


class DynamicConversationalSwarm:
    def __init__(
        self,
        name: str = "Dynamic Conversational Swarm",
        description: str = "A swarm that uses a dynamic conversational model to solve complex tasks.",
        agents: List[Agent] = [],
        max_loops: int = 1,
        output_type: str = "list",
        shield_config: Optional[ShieldConfig] = None,
        enable_security: bool = True,
        security_level: str = "standard",
        *args,
        **kwargs,
    ):
        self.name = name
        self.description = description
        self.agents = agents
        self.max_loops = max_loops
        self.output_type = output_type

        # Initialize SwarmShield integration
        self._initialize_swarm_shield(shield_config, enable_security, security_level)

        self.conversation = Conversation()

        # Agents in the chat
        agents_in_chat = self.get_agents_info()
        self.conversation.add(
            role="Conversation Log", content=agents_in_chat
        )

        self.inject_tools()

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

    # Inject tools into the agents
    def inject_tools(self):
        for agent in self.agents:
            agent.tools_list_dictionary = tools

    def parse_json_into_dict(self, json_str: str) -> dict:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON string")

    def run_agent(self, agent_name: str, task: str) -> str:
        """
        Run a specific agent with a given task.

        Args:
            agent_name (str): The name of the agent to run
            task (str): The task to execute

        Returns:
            str: The agent's response to the task

        Raises:
            ValueError: If agent is not found
            RuntimeError: If there's an error running the agent
        """
        agent = find_agent_by_name(
            agents=self.agents, agent_name=agent_name
        )
        return agent.run(task)

    def fetch_random_agent_name(self) -> str:
        return random.choice(self.agents).agent_name

    def run(self, task: str) -> str:
        """
        Run the dynamic conversational swarm for a specified number of loops.
        Each agent has access to the full conversation history.

        Args:
            task (str): The initial task/prompt to process

        Returns:
            str: The final response after all loops are complete
        """
        self.conversation.add(
            role=f"{self.fetch_random_agent_name()}", content=task
        )

        # for loop in range(self.max_loops):
        #     # Add loop marker to conversation for clarity
        #     self.conversation.add(
        #         role="System",
        #         content=f"=== Starting Loop {loop + 1}/{self.max_loops} ==="
        #     )

        #     # First agent interaction
        #     current_agent = self.randomly_select_agent()
        #     response = self.run_agent(current_agent.name, self.conversation.get_str())
        #     self.conversation.add(role=current_agent.name, content=any_to_str(response))

        #     try:
        #         # Parse response and get next agent
        #         response_dict = self.parse_json_into_dict(response)

        #         # Check if we should continue or end the loop
        #         if not response_dict.get("respond_or_no_respond", True):
        #             break

        #         # Get the task description for the next agent
        #         next_task = response_dict.get("task_description", self.conversation.get_str())

        #         # Run the next agent with the specific task description
        #         next_agent = self.find_agent_by_name(response_dict["agent_name"])
        #         next_response = self.run_agent(next_agent.name, next_task)

        #         # Add both the task description and response to the conversation
        #         self.conversation.add(
        #             role="System",
        #             content=f"Response from {response_dict['agent_name']}: {next_task}"
        #         )
        #         self.conversation.add(role=next_agent.name, content=any_to_str(next_response))

        #     except (ValueError, KeyError) as e:
        #         self.conversation.add(
        #             role="System",
        #             content=f"Error in loop {loop + 1}: {str(e)}"
        #         )
        #         break

        # Run first agent
        current_agent = self.randomly_select_agent()
        response = self.run_agent(
            current_agent.agent_name, self.conversation.get_str()
        )
        self.conversation.add(
            role=current_agent.agent_name,
            content=any_to_str(response),
        )

        # Convert to json
        response_dict = self.parse_json_into_dict(response)

        # Fetch task
        respone_two = response_dict["response"]
        agent_name = response_dict["agent_name"]

        print(f"Response from {agent_name}: {respone_two}")

        # Run next agent
        next_response = self.run_agent(
            agent_name, self.conversation.get_str()
        )
        self.conversation.add(
            role=agent_name, content=any_to_str(next_response)
        )

        # # Get the next agent
        # response_three = self.parse_json_into_dict(next_response)
        # agent_name_three = response_three["agent_name"]
        # respone_four = response_three["response"]

        # print(f"Response from {agent_name_three}: {respone_four}")
        # # Run the next agent
        # next_response = self.run_agent(agent_name_three, self.conversation.get_str())
        # self.conversation.add(role=agent_name_three, content=any_to_str(next_response))

        # Format and return the final conversation history
        return history_output_formatter(
            self.conversation, type=self.output_type
        )

    def randomly_select_agent(self) -> Agent:
        return random.choice(self.agents)

    def get_agents_info(self) -> str:
        """
        Fetches and formats information about all available agents in the system.

        Returns:
            str: A formatted string containing names and descriptions of all agents.
        """
        if not self.agents:
            return "No agents currently available in the system."

        agents_info = [
            "Agents In the System:",
            "",
        ]  # Empty string for line spacing

        for idx, agent in enumerate(self.agents, 1):
            agents_info.extend(
                [
                    f"[Agent {idx}]",
                    f"Name: {agent.name}",
                    f"Description: {agent.description}",
                    "",  # Empty string for line spacing between agents
                ]
            )

        return "\n".join(agents_info).strip()
