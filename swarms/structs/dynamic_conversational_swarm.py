import json
import random
from swarms.structs.agent import Agent
from typing import List
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.any_to_str import any_to_str

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
        *args,
        **kwargs,
    ):
        self.name = name
        self.description = description
        self.agents = agents
        self.max_loops = max_loops
        self.output_type = output_type

        self.conversation = Conversation()

        # Agents in the chat
        agents_in_chat = self.get_agents_info()
        self.conversation.add(
            role="Conversation Log", content=agents_in_chat
        )

        self.inject_tools()

    # Inject tools into the agents
    def inject_tools(self):
        for agent in self.agents:
            agent.tools_list_dictionary = tools

    def parse_json_into_dict(self, json_str: str) -> dict:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON string")

    def find_agent_by_name(self, agent_name: str) -> Agent:
        for agent in self.agents:
            if agent.name == agent_name:
                return agent
        raise ValueError(f"Agent with name {agent_name} not found")

    def run_agent(self, agent_name: str, task: str) -> str:
        agent = self.find_agent_by_name(agent_name)
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
