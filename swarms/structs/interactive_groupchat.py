import re
from typing import Callable, List, Union

from loguru import logger

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.generate_keys import generate_api_key
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)


class InteractiveGroupChatError(Exception):
    """Base exception class for InteractiveGroupChat errors"""

    pass


class AgentNotFoundError(InteractiveGroupChatError):
    """Raised when a mentioned agent is not found in the group"""

    pass


class NoMentionedAgentsError(InteractiveGroupChatError):
    """Raised when no agents are mentioned in the task"""

    pass


class InvalidTaskFormatError(InteractiveGroupChatError):
    """Raised when the task format is invalid"""

    pass


class InteractiveGroupChat:
    """
    An interactive group chat system that enables conversations with multiple agents using @mentions.

    This class allows users to interact with multiple agents by mentioning them using @agent_name syntax.
    When multiple agents are mentioned, they can see and respond to each other's tasks.

    Attributes:
        name (str): Name of the group chat
        description (str): Description of the group chat's purpose
        agents (List[Union[Agent, Callable]]): List of Agent instances or callable functions
        max_loops (int): Maximum number of conversation turns
        conversation (Conversation): Stores the chat history
        agent_map (Dict[str, Union[Agent, Callable]]): Mapping of agent names to their instances

    Args:
        name (str, optional): Name of the group chat. Defaults to "InteractiveGroupChat".
        description (str, optional): Description of the chat. Defaults to "An interactive group chat for multiple agents".
        agents (List[Union[Agent, Callable]], optional): List of participating agents or callables. Defaults to empty list.
        max_loops (int, optional): Maximum conversation turns. Defaults to 1.
        output_type (str, optional): Type of output format. Defaults to "string".
        interactive (bool, optional): Whether to enable interactive terminal mode. Defaults to False.

    Raises:
        ValueError: If invalid initialization parameters are provided
    """

    def __init__(
        self,
        id: str = generate_api_key(prefix="swarms-"),
        name: str = "InteractiveGroupChat",
        description: str = "An interactive group chat for multiple agents",
        agents: List[Union[Agent, Callable]] = [],
        max_loops: int = 1,
        output_type: str = "string",
        interactive: bool = False,
    ):
        self.id = id
        self.name = name
        self.description = description
        self.agents = agents
        self.max_loops = max_loops
        self.output_type = output_type
        self.interactive = interactive

        # Initialize conversation history
        self.conversation = Conversation(time_enabled=True)

        # Create a mapping of agent names to agents for easy lookup
        self.agent_map = {}
        for agent in agents:
            if isinstance(agent, Agent):
                self.agent_map[agent.agent_name] = agent
            elif callable(agent):
                # For callable functions, use the function name as the agent name
                self.agent_map[agent.__name__] = agent

        self._validate_initialization()
        self._setup_conversation_context()
        self._update_agent_prompts()

    def _validate_initialization(self) -> None:
        """
        Validates the group chat configuration.

        Raises:
            ValueError: If any required components are missing or invalid
        """
        if len(self.agents) < 1:
            raise ValueError(
                "At least one agent is required for the group chat"
            )

        if self.max_loops <= 0:
            raise ValueError("Max loops must be greater than 0")

    def _setup_conversation_context(self) -> None:
        """Sets up the initial conversation context with group chat information."""
        agent_info = []
        for agent in self.agents:
            if isinstance(agent, Agent):
                agent_info.append(
                    f"- {agent.agent_name}: {agent.system_prompt}"
                )
            elif callable(agent):
                agent_info.append(
                    f"- {agent.__name__}: Custom callable function"
                )

        context = (
            f"Group Chat Name: {self.name}\n"
            f"Description: {self.description}\n"
            f"Available Agents:\n" + "\n".join(agent_info)
        )
        self.conversation.add(role="System", content=context)

    def _update_agent_prompts(self) -> None:
        """Updates each agent's system prompt with information about other agents and the group chat."""
        agent_info = []
        for agent in self.agents:
            if isinstance(agent, Agent):
                agent_info.append(
                    {
                        "name": agent.agent_name,
                        "description": agent.system_prompt,
                    }
                )
            elif callable(agent):
                agent_info.append(
                    {
                        "name": agent.__name__,
                        "description": "Custom callable function",
                    }
                )

        group_context = (
            f"\n\nYou are part of a group chat named '{self.name}' with the following description: {self.description}\n"
            f"Other participants in this chat:\n"
        )

        for agent in self.agents:
            if isinstance(agent, Agent):
                # Create context excluding the current agent
                other_agents = [
                    info
                    for info in agent_info
                    if info["name"] != agent.agent_name
                ]
                agent_context = group_context
                for other in other_agents:
                    agent_context += (
                        f"- {other['name']}: {other['description']}\n"
                    )

                # Update the agent's system prompt
                agent.system_prompt = (
                    agent.system_prompt + agent_context
                )
                logger.info(
                    f"Updated system prompt for agent: {agent.agent_name}"
                )

    def _extract_mentions(self, task: str) -> List[str]:
        """
        Extracts @mentions from the task.

        Args:
            task (str): The input task

        Returns:
            List[str]: List of mentioned agent names

        Raises:
            InvalidtaskFormatError: If the task format is invalid
        """
        try:
            # Find all @mentions using regex
            mentions = re.findall(r"@(\w+)", task)
            return [
                mention
                for mention in mentions
                if mention in self.agent_map
            ]
        except Exception as e:
            logger.error(f"Error extracting mentions: {e}")
            raise InvalidTaskFormatError(f"Invalid task format: {e}")

    def start_interactive_session(self):
        """
        Start an interactive terminal session for chatting with agents.

        This method creates a REPL (Read-Eval-Print Loop) that allows users to:
        - Chat with agents using @mentions
        - See available agents and their descriptions
        - Exit the session using 'exit' or 'quit'
        - Get help using 'help' or '?'
        """
        if not self.interactive:
            raise InteractiveGroupChatError(
                "Interactive mode is not enabled. Initialize with interactive=True"
            )

        print(f"\nWelcome to {self.name}!")
        print(f"Description: {self.description}")
        print("\nAvailable agents:")
        for name, agent in self.agent_map.items():
            if isinstance(agent, Agent):
                print(
                    f"- @{name}: {agent.system_prompt.splitlines()[0]}"
                )
            else:
                print(f"- @{name}: Custom callable function")

        print("\nCommands:")
        print("- Type 'help' or '?' for help")
        print("- Type 'exit' or 'quit' to end the session")
        print("- Use @agent_name to mention agents")
        print("\nStart chatting:")

        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()

                # Handle special commands
                if user_input.lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    break

                if user_input.lower() in ["help", "?"]:
                    print("\nHelp:")
                    print("1. Mention agents using @agent_name")
                    print(
                        "2. You can mention multiple agents in one task"
                    )
                    print("3. Available agents:")
                    for name in self.agent_map:
                        print(f"   - @{name}")
                    print(
                        "4. Type 'exit' or 'quit' to end the session"
                    )
                    continue

                if not user_input:
                    continue

                # Process the task and get responses
                try:
                    response = self.run(user_input)
                    print("\nChat:")
                    print(response)

                except NoMentionedAgentsError:
                    print(
                        "\nError: Please mention at least one agent using @agent_name"
                    )
                except AgentNotFoundError as e:
                    print(f"\nError: {str(e)}")
                except Exception as e:
                    print(f"\nAn error occurred: {str(e)}")

            except KeyboardInterrupt:
                print("\nSession terminated by user. Goodbye!")
                break
            except Exception as e:
                print(f"\nAn unexpected error occurred: {str(e)}")
                print(
                    "The session will continue. You can type 'exit' to end it."
                )

    def run(self, task: str) -> str:
        """
        Process a task and get responses from mentioned agents.
        If interactive mode is enabled, this will be called by start_interactive_session().
        Otherwise, it can be called directly for single task processing.
        """
        try:
            # Extract mentioned agents
            mentioned_agents = self._extract_mentions(task)

            if not mentioned_agents:
                raise NoMentionedAgentsError(
                    "No valid agents mentioned in the task"
                )

            # Add user task to conversation
            self.conversation.add(role="User", content=task)

            # Get responses from mentioned agents
            for agent_name in mentioned_agents:
                agent = self.agent_map.get(agent_name)
                if not agent:
                    raise AgentNotFoundError(
                        f"Agent '{agent_name}' not found"
                    )

                try:
                    # Get the complete conversation history
                    context = (
                        self.conversation.return_history_as_string()
                    )

                    # Get response from agent
                    if isinstance(agent, Agent):
                        response = agent.run(
                            task=f"{context}\nPlease respond to the latest task as {agent_name}."
                        )
                    else:
                        # For callable functions
                        response = agent(context)

                    # Add response to conversation
                    if response and not response.isspace():
                        self.conversation.add(
                            role=agent_name, content=response
                        )
                        logger.info(f"Agent {agent_name} responded")

                except Exception as e:
                    logger.error(
                        f"Error getting response from {agent_name}: {e}"
                    )
                    self.conversation.add(
                        role=agent_name,
                        content=f"Error: Unable to generate response - {str(e)}",
                    )

            return history_output_formatter(
                self.conversation, self.output_type
            )

        except InteractiveGroupChatError as e:
            logger.error(f"GroupChat error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise InteractiveGroupChatError(
                f"Unexpected error occurred: {str(e)}"
            )
