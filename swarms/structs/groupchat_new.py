import os
import asyncio
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from swarms import Agent
from dotenv import load_dotenv
from swarms.utils.formatter import formatter

# Load environment variables
load_dotenv()

# Get OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")


# Define Pydantic schema for agent outputs
class AgentOutput(BaseModel):
    """Schema for capturing the output of each agent."""

    agent_name: str = Field(..., description="The name of the agent")
    message: str = Field(
        ...,
        description="The agent's response or contribution to the group chat",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the agent's response",
    )


class GroupChat:
    """
    GroupChat class to enable multiple agents to communicate in an asynchronous group chat.
    Each agent is aware of all other agents, every message exchanged, and the social context.
    """

    def __init__(
        self,
        name: str,
        description: str,
        agents: List[Agent],
        max_loops: int = 1,
    ):
        """
        Initialize the GroupChat.

        Args:
            name (str): Name of the group chat.
            description (str): Description of the purpose of the group chat.
            agents (List[Agent]): A list of agents participating in the chat.
            max_loops (int): Maximum number of loops to run through all agents.
        """
        self.name = name
        self.description = description
        self.agents = agents
        self.max_loops = max_loops
        self.chat_history = (
            []
        )  # Stores all messages exchanged in the chat

        formatter.print_panel(
            f"Initialized GroupChat '{self.name}' with {len(self.agents)} agents. Max loops: {self.max_loops}",
            title="Groupchat Swarm",
        )

    async def _agent_conversation(
        self, agent: Agent, input_message: str
    ) -> AgentOutput:
        """
        Facilitate a single agent's response to the chat.

        Args:
            agent (Agent): The agent responding.
            input_message (str): The message triggering the response.

        Returns:
            AgentOutput: The agent's response captured in a structured format.
        """
        formatter.print_panel(
            f"Agent '{agent.agent_name}' is responding to the message: {input_message}",
            title="Groupchat Swarm",
        )
        response = await asyncio.to_thread(agent.run, input_message)

        output = AgentOutput(
            agent_name=agent.agent_name,
            message=response,
            metadata={"context_length": agent.context_length},
        )
        # logger.debug(f"Agent '{agent.agent_name}' response: {response}")
        return output

    async def _run(self, initial_message: str) -> List[AgentOutput]:
        """
        Execute the group chat asynchronously, looping through all agents up to max_loops.

        Args:
            initial_message (str): The initial message to start the chat.

        Returns:
            List[AgentOutput]: The responses of all agents across all loops.
        """
        formatter.print_panel(
            f"Starting group chat '{self.name}' with initial message: {initial_message}",
            title="Groupchat Swarm",
        )
        self.chat_history.append(
            {"sender": "System", "message": initial_message}
        )

        outputs = []
        for loop in range(self.max_loops):
            formatter.print_panel(
                f"Group chat loop {loop + 1}/{self.max_loops}",
                title="Groupchat Swarm",
            )

            for agent in self.agents:
                # Create a custom input message for each agent, sharing the chat history and social context
                input_message = (
                    f"Chat History:\n{self._format_chat_history()}\n\n"
                    f"Participants:\n"
                    + "\n".join(
                        [
                            f"- {a.agent_name}: {a.system_prompt}"
                            for a in self.agents
                        ]
                    )
                    + f"\n\nNew Message: {initial_message}\n\n"
                    f"You are '{agent.agent_name}'. Remember to keep track of the social context, who is speaking, "
                    f"and respond accordingly based on your role: {agent.system_prompt}."
                )

                # Collect agent's response
                output = await self._agent_conversation(
                    agent, input_message
                )
                outputs.append(output)

                # Update chat history with the agent's response
                self.chat_history.append(
                    {
                        "sender": agent.agent_name,
                        "message": output.message,
                    }
                )

        formatter.print_panel(
            "Group chat completed. All agent responses captured.",
            title="Groupchat Swarm",
        )
        return outputs

    def run(self, task: str, *args, **kwargs):
        return asyncio.run(self.run(task, *args, **kwargs))

    def _format_chat_history(self) -> str:
        """
        Format the chat history for agents to understand the context.

        Returns:
            str: The formatted chat history as a string.
        """
        return "\n".join(
            [
                f"{entry['sender']}: {entry['message']}"
                for entry in self.chat_history
            ]
        )

    def __str__(self) -> str:
        """String representation of the group chat's outputs."""
        return self._format_chat_history()

    def to_json(self) -> str:
        """JSON representation of the group chat's outputs."""
        return [
            {"sender": entry["sender"], "message": entry["message"]}
            for entry in self.chat_history
        ]


# # Example Usage
# if __name__ == "__main__":

#     load_dotenv()

#     # Get the OpenAI API key from the environment variable
#     api_key = os.getenv("OPENAI_API_KEY")

#     # Create an instance of the OpenAIChat class
#     model = OpenAIChat(
#         openai_api_key=api_key,
#         model_name="gpt-4o-mini",
#         temperature=0.1,
#     )

#     # Example agents
#     agent1 = Agent(
#         agent_name="Financial-Analysis-Agent",
#         system_prompt="You are a financial analyst specializing in investment strategies.",
#         llm=model,
#         max_loops=1,
#         autosave=False,
#         dashboard=False,
#         verbose=True,
#         dynamic_temperature_enabled=True,
#         user_name="swarms_corp",
#         retry_attempts=1,
#         context_length=200000,
#         output_type="string",
#         streaming_on=False,
#     )

#     agent2 = Agent(
#         agent_name="Tax-Adviser-Agent",
#         system_prompt="You are a tax adviser who provides clear and concise guidance on tax-related queries.",
#         llm=model,
#         max_loops=1,
#         autosave=False,
#         dashboard=False,
#         verbose=True,
#         dynamic_temperature_enabled=True,
#         user_name="swarms_corp",
#         retry_attempts=1,
#         context_length=200000,
#         output_type="string",
#         streaming_on=False,
#     )

#     # Create group chat
#     group_chat = GroupChat(
#         name="Financial Discussion",
#         description="A group chat for financial analysis and tax advice.",
#         agents=[agent1, agent2],
#     )

#     # Run the group chat
#     asyncio.run(
#         group_chat.run(
#             "How can I establish a ROTH IRA to buy stocks and get a tax break? What are the criteria? What do you guys think?"
#         )
#     )
