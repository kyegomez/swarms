import concurrent.futures
import random
import re
import traceback
from typing import Callable, List, Optional, Union

from loguru import logger

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.ma_utils import create_agent_map
from swarms.utils.generate_keys import generate_api_key
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.prompts.multi_agent_collab_prompt import (
    MULTI_AGENT_COLLAB_PROMPT_TWO,
)

SpeakerFunction = Callable[[List[str], "Agent"], bool]


def get_mention_instruction() -> str:
    """
    Returns the instruction prompt for teaching agents how to use @mentions.

    Returns:
        str: The mention instruction prompt text
    """
    return """

    IMPORTANT: You are part of a collaborative group chat where you can interact with other agents using @mentions.

    -COLLABORATIVE RESPONSE PROTOCOL:
    1. FIRST: Read and understand all previous responses from other agents
    2. ACKNOWLEDGE: Reference and acknowledge what other agents have said
    3. BUILD UPON: Add your perspective while building upon their insights
    4. MENTION: Use @agent_name to call on other agents when needed
    5. COMPLETE: Acknowledge when your part is done and what still needs to be done

    HOW TO MENTION OTHER AGENTS:
    - Use @agent_name to mention another agent in your response
    - You can mention multiple agents: @agent1 @agent2
    - When you mention an agent, they will be notified and can respond
    - Example: "I think @analyst should review this data" or "Let's ask @researcher to investigate this further"

    AVAILABLE AGENTS TO MENTION:
    """


def get_group_context(name: str, description: str) -> str:
    """
    Returns the group context prompt with name and description.

    Args:
        name (str): Name of the group chat
        description (str): Description of the group chat

    Returns:
        str: The group context prompt text
    """
    return (
        f"\n\nYou are part of a group chat named '{name}' with the following description: {description}\n"
        f"Other participants in this chat:\n"
    )


def get_collaboration_guidelines() -> str:
    """
    Returns the collaboration guidelines prompt.

    Returns:
        str: The collaboration guidelines prompt text
    """
    return """

    COLLABORATION GUIDELINES:
    - ALWAYS read the full conversation history before responding
    - ACKNOWLEDGE other agents' contributions: "Building on @analyst's data insights..." or "I agree with @researcher's findings that..."
    - BUILD UPON previous responses rather than repeating information
    - SYNTHESIZE multiple perspectives when possible
    - ASK CLARIFYING QUESTIONS if you need more information from other agents
    - DELEGATE appropriately: "Let me ask @expert_agent to verify this" or "@specialist, can you elaborate on this point?"

    TASK COMPLETION GUIDELINES:
    - ACKNOWLEDGE when you are done with your part of the task
    - Clearly state what still needs to be done before the overall task is finished
    - If you mention other agents, explain what specific input you need from them
    - Use phrases like "I have completed [specific part]" or "The task still requires [specific actions]"

    RESPONSE STRUCTURE:
    1. ACKNOWLEDGE: "I've reviewed the responses from @agent1 and @agent2..."
    2. BUILD: "Building on @agent1's analysis of the data..."
    3. CONTRIBUTE: "From my perspective, I would add..."
    4. COLLABORATE: "To get a complete picture, let me ask @agent3 to..."
    5. COMPLETE: "I have completed [my part]. The task still requires [specific next steps]"
    6. SYNTHESIZE: "Combining our insights, the key findings are..."

    EXAMPLES OF GOOD COLLABORATION:
    - "I've reviewed @analyst's data analysis and @researcher's market insights. The data shows strong growth potential, and I agree with @researcher that we should focus on emerging markets. Let me add that from a content perspective, we should @writer to create targeted messaging for these markets. I have completed my market analysis. The task now requires @writer to develop content and @reviewer to validate the approach."
    - "Building on @researcher's findings about customer behavior, I can see that @analyst's data supports this trend. To get a complete understanding, let me ask @writer to help us craft messaging that addresses these specific customer needs. My data analysis is complete. The task still needs @writer to create messaging and @reviewer to approve the final strategy."

    AVOID:
    - Ignoring other agents' responses
    - Repeating what others have already said
    - Making assumptions without consulting relevant experts
    - Responding in isolation without considering the group's collective knowledge
    - Not acknowledging task completion status

    Remember: You are part of a team. Your response should reflect that you've read, understood, and are building upon the contributions of others, and clearly communicate your task completion status.
    """


def get_agent_context_prompt(
    name: str, description: str, other_agents: List[dict]
) -> str:
    """
    Returns the complete agent context prompt with group info and other agents.

    Args:
        name (str): Name of the group chat
        description (str): Description of the group chat
        other_agents (List[dict]): List of dictionaries with 'name' and 'description' keys for other agents

    Returns:
        str: The complete agent context prompt text
    """
    group_context = get_group_context(name, description)
    mention_instruction = get_mention_instruction()
    collaboration_guidelines = get_collaboration_guidelines()

    agent_context = group_context + mention_instruction
    for other in other_agents:
        agent_context += (
            f"- @{other['name']}: {other['description']}\n"
        )

    agent_context += collaboration_guidelines
    return agent_context


def get_collaborative_task_prompt(
    context: str, agent_name: str
) -> str:
    """
    Returns the collaborative task prompt for agent responses.

    Args:
        context (str): The conversation history context
        agent_name (str): Name of the agent responding

    Returns:
        str: The collaborative task prompt text
    """
    return f"""{context}

COLLABORATIVE TASK: Please respond to the latest task as {agent_name}.

IMPORTANT INSTRUCTIONS:
1. Read the ENTIRE conversation history above
2. Acknowledge what other agents have said before adding your perspective
3. Build upon their insights rather than repeating information
4. If you need input from other agents, mention them using @agent_name
5. Provide your unique expertise while showing you understand the group's collective knowledge

TASK COMPLETION GUIDELINES:
- Acknowledge when you are done with your part of the task
- Clearly state what still needs to be done before the overall task is finished
- If you mention other agents, explain what specific input you need from them
- Use phrases like "I have completed [specific part]" or "The task still requires [specific actions]"

Remember: You are part of a collaborative team. Your response should demonstrate that you've read, understood, and are building upon the contributions of others."""


class GroupChatError(Exception):
    """Base exception class for GroupChat errors"""

    pass


class AgentNotFoundError(GroupChatError):
    """Raised when a mentioned agent is not found in the group"""

    pass


class InvalidTaskFormatError(GroupChatError):
    """Raised when the task format is invalid"""

    pass


class InvalidSpeakerFunctionError(GroupChatError):
    """Raised when an invalid speaker function is provided"""

    pass


# Built-in speaker functions
def round_robin_speaker(
    agents: List[str], current_index: int = 0
) -> str:
    """
    Round robin speaker function that cycles through agents in order.

    Args:
        agents: List of agent names
        current_index: Current position in the cycle

    Returns:
        Next agent name in the round robin sequence
    """
    if not agents:
        raise ValueError("No agents provided for round robin")
    return agents[current_index % len(agents)]


def random_speaker(agents: List[str], **kwargs) -> str:
    """
    Random speaker function that selects agents randomly.

    Args:
        agents: List of agent names
        **kwargs: Additional arguments (ignored)

    Returns:
        Randomly selected agent name
    """
    if not agents:
        raise ValueError("No agents provided for random selection")
    return random.choice(agents)


def priority_speaker(
    agents: List[str], priorities: dict, **kwargs
) -> str:
    """
    Priority-based speaker function that selects agents based on priority weights.

    Args:
        agents: List of agent names
        priorities: Dictionary mapping agent names to priority weights
        **kwargs: Additional arguments (ignored)

    Returns:
        Selected agent name based on priority weights
    """
    if not agents:
        raise ValueError("No agents provided for priority selection")

    # Filter agents that exist in the priorities dict
    available_agents = [
        agent for agent in agents if agent in priorities
    ]
    if not available_agents:
        # Fallback to random if no priorities match
        return random.choice(agents)

    # Calculate total weight
    total_weight = sum(
        priorities[agent] for agent in available_agents
    )
    if total_weight == 0:
        return random.choice(available_agents)

    # Select based on weighted probability
    rand_val = random.uniform(0, total_weight)
    current_weight = 0

    for agent in available_agents:
        current_weight += priorities[agent]
        if rand_val <= current_weight:
            return agent

    return available_agents[-1]  # Fallback


def random_dynamic_speaker(
    agents: List[str],
    response: str = "",
    strategy: str = "parallel",
    **kwargs,
) -> Union[str, List[str]]:
    """
    Random dynamic speaker function that selects agents based on @mentions in responses.

    This function works in two phases:
    1. If no response is provided (first call), randomly selects an agent
    2. If a response is provided, extracts @mentions and returns agent(s) based on strategy

    Args:
        agents: List of available agent names
        response: The response from the previous agent (may contain @mentions)
        strategy: How to handle multiple mentions - "sequential" or "parallel"
        **kwargs: Additional arguments (ignored)

    Returns:
        For sequential strategy: str (single agent name)
        For parallel strategy: List[str] (list of agent names)
    """
    if not agents:
        raise ValueError(
            "No agents provided for random dynamic selection"
        )

    # If no response provided, randomly select first agent
    if not response:
        return random.choice(agents)

    # Extract @mentions from the response
    mentions = re.findall(r"@(\w+)", response)

    # Filter mentions to only include valid agents
    valid_mentions = [
        mention for mention in mentions if mention in agents
    ]

    if not valid_mentions:
        # If no valid mentions, randomly select from all agents
        return random.choice(agents)

    # Handle multiple mentions based on strategy
    if strategy == "sequential":
        # Return the first mentioned agent for sequential execution
        return valid_mentions[0]
    elif strategy == "parallel":
        # Return all mentioned agents for parallel execution
        return valid_mentions
    else:
        raise ValueError(
            f"Invalid strategy: {strategy}. Must be 'sequential' or 'parallel'"
        )


speaker_functions = {
    "round-robin-speaker": round_robin_speaker,
    "random-speaker": random_speaker,
    "priority-speaker": priority_speaker,
    "random-dynamic-speaker": random_dynamic_speaker,
}


def round_robin(history: List[str], agent: Agent) -> bool:
    """
    Round robin speaker function.
    Each agent speaks in turn, in a circular order.
    """
    return True


def expertise_based(history: List[str], agent: Agent) -> bool:
    """
    Expertise based speaker function.
    An agent speaks if their system prompt is in the last message.
    """
    return (
        agent.system_prompt.lower() in history[-1].lower()
        if history
        else True
    )


def random_selection(history: List[str], agent: Agent) -> bool:
    """
    Random selection speaker function.
    An agent speaks randomly.
    """
    import random

    return random.choice([True, False])


def custom_speaker(history: List[str], agent: Agent) -> bool:
    """
    Custom speaker function with complex logic.

    Args:
        history: Previous conversation messages
        agent: Current agent being evaluated

    Returns:
        bool: Whether agent should speak
    """
    # No history - let everyone speak
    if not history:
        return True

    last_message = history[-1].lower()

    # Check for agent expertise keywords
    expertise_relevant = any(
        keyword in last_message
        for keyword in agent.description.lower().split()
    )

    # Check for direct mentions
    mentioned = agent.agent_name.lower() in last_message

    # Check if agent hasn't spoken recently
    not_recent_speaker = not any(
        agent.agent_name in msg for msg in history[-3:]
    )

    return expertise_relevant or mentioned or not_recent_speaker


def most_recent(history: List[str], agent: Agent) -> bool:
    """
    Most recent speaker function.
    An agent speaks if they are the last speaker.
    """
    return (
        agent.agent_name == history[-1].split(":")[0].strip()
        if history
        else True
    )


def sentiment_based(history: List[str], agent: Agent) -> bool:
    """
    Sentiment based speaker function.
    An agent speaks if the last message has a sentiment matching their personality.
    """
    if not history:
        return True

    last_message = history[-1].lower()
    positive_words = [
        "good",
        "great",
        "excellent",
        "happy",
        "positive",
    ]
    negative_words = [
        "bad",
        "poor",
        "terrible",
        "unhappy",
        "negative",
    ]

    is_positive = any(word in last_message for word in positive_words)
    is_negative = any(word in last_message for word in negative_words)

    # Assuming agent has a "personality" trait in description
    agent_is_positive = "positive" in agent.description.lower()

    return is_positive if agent_is_positive else is_negative


def length_based(history: List[str], agent: Agent) -> bool:
    """
    Length based speaker function.
    An agent speaks if the last message is longer/shorter than a threshold.
    """
    if not history:
        return True

    last_message = history[-1]
    threshold = 100

    # Some agents prefer long messages, others short
    prefers_long = "detailed" in agent.description.lower()
    message_is_long = len(last_message) > threshold

    return message_is_long if prefers_long else not message_is_long


def question_based(history: List[str], agent: Agent) -> bool:
    """
    Question based speaker function.
    An agent speaks if the last message contains a question.
    """
    if not history:
        return True

    last_message = history[-1]
    question_indicators = [
        "?",
        "what",
        "how",
        "why",
        "when",
        "where",
        "who",
    ]

    return any(
        indicator in last_message.lower()
        for indicator in question_indicators
    )


def topic_based(history: List[str], agent: Agent) -> bool:
    """
    Topic based speaker function.
    An agent speaks if their expertise matches the current conversation topic.
    """
    if not history:
        return True

    # Look at last 3 messages to determine topic
    recent_messages = history[-3:] if len(history) >= 3 else history
    combined_text = " ".join(msg.lower() for msg in recent_messages)

    # Extract expertise topics from agent description
    expertise_topics = [
        word.lower()
        for word in agent.description.split()
        if len(word) > 4
    ]  # Simple topic extraction

    return any(topic in combined_text for topic in expertise_topics)


class GroupChat:
    """
    GroupChat enables collaborative conversations among multiple agents (or callables)
    with flexible speaker selection strategies, conversation history, and interactive terminal sessions.

    Features:
        - Supports multiple agents (LLMs or callable functions)
        - Customizable speaker selection (round robin, random, priority, dynamic, or custom)
        - Maintains conversation history with time stamps
        - Interactive REPL session for human-in-the-loop group chat
        - Agents can @mention each other to request input or delegate tasks
        - Automatic prompt augmentation to teach agents collaborative protocols

    Args:
        id (str): Unique identifier for the group chat (default: generated)
        name (str): Name of the group chat
        description (str): Description of the group chat's purpose
        agents (List[Union[Agent, Callable]]): List of agent objects or callables
        max_loops (int): Maximum number of conversation loops per run
        output_type (str): Output format for conversation history ("dict", "str", etc.)
        interactive (bool): If True, enables interactive terminal session
        speaker_function (Optional[Union[str, Callable]]): Speaker selection strategy
        speaker_state (Optional[dict]): State/config for the speaker function

    Raises:
        ValueError: If required arguments are missing or invalid
        InvalidSpeakerFunctionError: If an invalid speaker function is provided
        GroupChatError: For interactive session errors
        AgentNotFoundError: If an agent is not found by name
    """

    def __init__(
        self,
        id: str = generate_api_key(prefix="swarms-"),
        name: str = "GroupChat",
        description: str = "A group chat for multiple agents",
        agents: List[Union[Agent, Callable]] = [],
        max_loops: int = 1,
        output_type: str = "dict",
        interactive: bool = False,
        speaker_function: Optional[Union[str, Callable]] = None,
        speaker_state: Optional[dict] = None,
        # Legacy parameters for backward compatibility
        speaker_fn: Optional[SpeakerFunction] = None,
        rules: str = "",
        time_enabled: bool = True,
    ):
        """
        Initialize the GroupChat.

        Args:
            id (str): Unique identifier for the group chat.
            name (str): Name of the group chat.
            description (str): Description of the group chat.
            agents (List[Union[Agent, Callable]]): List of agent objects or callables.
            max_loops (int): Maximum number of conversation loops per run.
            output_type (str): Output format for conversation history.
            interactive (bool): If True, enables interactive terminal session.
            speaker_function (Optional[Union[str, Callable]]): Speaker selection strategy.
            speaker_state (Optional[dict]): State/config for the speaker function.
            speaker_fn (Optional[SpeakerFunction]): Legacy speaker function for backward compatibility.
            rules (str): Rules for the conversation.
            time_enabled (bool): Whether to enable timestamps in conversation.
        """
        self.id = id
        self.name = name
        self.description = description
        self.agents = agents
        self.max_loops = max_loops
        self.output_type = output_type
        self.interactive = interactive
        self.speaker_function = (
            speaker_function or speaker_fn
        )  # Support legacy parameter
        self.speaker_state = speaker_state
        self.rules = rules
        self.time_enabled = time_enabled

        self.setup()

    def _setup_speaker_function(self):
        # Speaker function configuration
        if self.speaker_function is None:
            self.speaker_function = round_robin_speaker
        elif isinstance(self.speaker_function, str):
            if self.speaker_function not in speaker_functions:
                available_functions = ", ".join(
                    speaker_functions.keys()
                )
                raise InvalidSpeakerFunctionError(
                    f"Invalid speaker function: '{self.speaker_function}'. "
                    f"Available functions: {available_functions}"
                )
            self.speaker_function = speaker_functions[
                self.speaker_function
            ]
        elif callable(self.speaker_function):
            self.speaker_function = self.speaker_function
        else:
            raise InvalidSpeakerFunctionError(
                "Speaker function must be either a string, callable, or None"
            )

        self.speaker_state = self.speaker_state or {
            "current_index": 0
        }

    def setup(self):
        """
        Set up the group chat, including speaker function, conversation history,
        agent mapping, and prompt augmentation.
        """

        # Initialize conversation history
        self.conversation = Conversation(
            time_enabled=self.time_enabled, rules=self.rules
        )

        self._setup_speaker_function()

        self.agent_map = create_agent_map(self.agents)

        self._validate_initialization()
        self._setup_conversation_context()
        self._update_agent_prompts()

    def set_speaker_function(
        self,
        speaker_function: Union[str, Callable],
        speaker_state: Optional[dict] = None,
    ) -> None:
        """
        Set the speaker function using either a string name or a custom callable.

        Args:
            speaker_function: Either a string name of a predefined function or a custom callable
                String options:
                - "round-robin-speaker": Cycles through agents in order
                - "random-speaker": Selects agents randomly
                - "priority-speaker": Selects based on priority weights
                - "random-dynamic-speaker": Randomly selects first agent, then follows @mentions in responses
                Callable: Custom function that takes (agents: List[str], **kwargs) -> str
            speaker_state: Optional state for the speaker function

        Raises:
            InvalidSpeakerFunctionError: If the speaker function is invalid
        """
        if isinstance(speaker_function, str):
            # Handle string-based speaker function
            if speaker_function not in speaker_functions:
                available_functions = ", ".join(
                    speaker_functions.keys()
                )
                raise InvalidSpeakerFunctionError(
                    f"Invalid speaker function: '{speaker_function}'. "
                    f"Available functions: {available_functions}"
                )
            self.speaker_function = speaker_functions[
                speaker_function
            ]
            logger.info(
                f"Speaker function set to: {speaker_function}"
            )
        elif callable(speaker_function):
            # Handle callable speaker function
            self.speaker_function = speaker_function
            logger.info(
                f"Custom speaker function set to: {speaker_function.__name__}"
            )
        else:
            raise InvalidSpeakerFunctionError(
                "Speaker function must be either a string or a callable"
            )

        # Update speaker state if provided
        if speaker_state:
            self.speaker_state.update(speaker_state)

        # Validate the speaker function
        self._validate_speaker_function()

    def get_available_speaker_functions(self) -> List[str]:
        """
        Get a list of available speaker function names.

        Returns:
            List[str]: List of available speaker function names
        """
        return list(speaker_functions.keys())

    def get_current_speaker_function(self) -> str:
        """
        Get the name of the current speaker function.

        Returns:
            str: Name of the current speaker function, or "custom" if it's a custom function
        """
        for name, func in speaker_functions.items():
            if self.speaker_function == func:
                return name
        return "custom"

    def start_interactive_session(self):
        """
        Start an interactive terminal session for chatting with agents.

        This method creates a REPL (Read-Eval-Print Loop) that allows users to:
        - Chat with agents using @mentions (optional)
        - See available agents and their descriptions
        - Exit the session using 'exit' or 'quit'
        - Get help using 'help' or '?'
        """
        if not self.interactive:
            raise GroupChatError(
                "Interactive mode is not enabled. Initialize with interactive=True"
            )

        print(f"\nWelcome to {self.name}!")
        print(f"Description: {self.description}")
        print(
            f"Current speaker function: {self.get_current_speaker_function()}"
        )
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
        print("- Type 'speaker' to change speaker function")
        print(
            "- Use @agent_name to mention specific agents (optional)"
        )
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
                    print(
                        "1. You can mention specific agents using @agent_name (optional)"
                    )
                    print(
                        "2. If no agents are mentioned, they will be selected automatically"
                    )
                    print("3. Available agents:")
                    for name in self.agent_map:
                        print(f"   - @{name}")
                    print(
                        "4. Type 'speaker' to change speaker function"
                    )
                    print(
                        "5. Type 'exit' or 'quit' to end the session"
                    )
                    continue

                if user_input.lower() == "speaker":
                    print(
                        f"\nCurrent speaker function: {self.get_current_speaker_function()}"
                    )
                    print("Available speaker functions:")
                    for i, func_name in enumerate(
                        self.get_available_speaker_functions(), 1
                    ):
                        print(f"  {i}. {func_name}")

                    try:
                        choice = input(
                            "\nEnter the number or name of the speaker function: "
                        ).strip()

                        # Try to parse as number first
                        try:
                            func_index = int(choice) - 1
                            if (
                                0
                                <= func_index
                                < len(
                                    self.get_available_speaker_functions()
                                )
                            ):
                                selected_func = self.get_available_speaker_functions()[
                                    func_index
                                ]
                            else:
                                print(
                                    "Invalid number. Please try again."
                                )
                                continue
                        except ValueError:
                            # Try to parse as name
                            selected_func = choice

                        self.set_speaker_function(selected_func)
                        print(
                            f"Speaker function changed to: {self.get_current_speaker_function()}"
                        )

                    except InvalidSpeakerFunctionError as e:
                        print(f"Error: {e}")
                    except Exception as e:
                        print(f"An error occurred: {e}")
                    continue

                if not user_input:
                    continue

                # Process the task and get responses
                try:
                    self.run(user_input)
                    print("\nChat:")
                    # print(response)

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
        """
        Sets up the initial conversation context with group chat information.
        Adds a system message describing the group and its agents.
        """
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
        """
        Updates each agent's system prompt with information about other agents and the group chat.
        This includes collaborative instructions and @mention usage guidelines.
        """
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

        for agent in self.agents:
            if isinstance(agent, Agent):
                # Create context excluding the current agent
                other_agents = [
                    info
                    for info in agent_info
                    if info["name"] != agent.agent_name
                ]
                agent_context = get_agent_context_prompt(
                    self.name, self.description, other_agents
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
        Extracts @mentions from the task. If no mentions are found, returns all available agents.

        Args:
            task (str): The input task

        Returns:
            List[str]: List of mentioned agent names or all agent names if no mentions

        Raises:
            InvalidTaskFormatError: If the task format is invalid
        """
        try:
            # Find all @mentions using regex
            mentions = re.findall(r"@(\w+)", task)
            valid_mentions = [
                mention
                for mention in mentions
                if mention in self.agent_map
            ]

            # If no valid mentions found, return all available agents
            if not valid_mentions:
                return list(self.agent_map.keys())

            return valid_mentions
        except Exception as e:
            logger.error(f"Error extracting mentions: {e}")
            raise InvalidTaskFormatError(f"Invalid task format: {e}")

    def _get_speaking_order(
        self, mentioned_agents: List[str]
    ) -> List[str]:
        """
        Determines the speaking order using the configured speaker function.

        Args:
            mentioned_agents: List of agent names that were mentioned

        Returns:
            List of agent names in the order they should speak
        """
        if not mentioned_agents:
            return []

        # Use the speaker function to determine order
        try:
            if self.speaker_function == round_robin_speaker:
                # For round robin, we need to maintain state
                current_index = self.speaker_state.get(
                    "current_index", 0
                )
                ordered_agents = []

                # Create the order starting from current index
                for i in range(len(mentioned_agents)):
                    agent = round_robin_speaker(
                        mentioned_agents, current_index + i
                    )
                    ordered_agents.append(agent)

                # Update state for next round
                self.speaker_state["current_index"] = (
                    current_index + len(mentioned_agents)
                ) % len(mentioned_agents)
                return ordered_agents

            elif self.speaker_function == random_speaker:
                # For random, shuffle the list
                shuffled = mentioned_agents.copy()
                random.shuffle(shuffled)
                return shuffled

            elif self.speaker_function == priority_speaker:
                # For priority, we need priorities in speaker_state
                priorities = self.speaker_state.get("priorities", {})
                if not priorities:
                    # Fallback to random if no priorities set
                    shuffled = mentioned_agents.copy()
                    random.shuffle(shuffled)
                    return shuffled

                # Sort by priority (higher priority first)
                sorted_agents = sorted(
                    mentioned_agents,
                    key=lambda x: priorities.get(x, 0),
                    reverse=True,
                )
                return sorted_agents

            elif self.speaker_function == random_dynamic_speaker:
                # For dynamic speaker, we need to handle it differently
                # The dynamic speaker will be called during the run method
                # For now, just return the original order
                return mentioned_agents

            else:
                # Custom speaker function
                # For custom functions, we'll use the first agent returned
                # and then process the rest in original order
                first_speaker = self.speaker_function(
                    mentioned_agents, **self.speaker_state
                )
                if first_speaker in mentioned_agents:
                    remaining = [
                        agent
                        for agent in mentioned_agents
                        if agent != first_speaker
                    ]
                    return [first_speaker] + remaining
                else:
                    return mentioned_agents

        except Exception as e:
            logger.error(f"Error in speaker function: {e}")
            # Fallback to original order
            return mentioned_agents

    def _process_dynamic_speakers(
        self,
        mentioned_agents: List[str],
        img: Optional[str],
        imgs: Optional[List[str]],
    ) -> None:
        """
        Process responses using the dynamic speaker function.

        Args:
            mentioned_agents (List[str]): List of agent names to consider for speaking.
            img (Optional[str]): Optional image input for the agents.
            imgs (Optional[List[str]]): Optional list of images for the agents.

        Returns:
            None
        """
        # Get strategy from speaker state (default to sequential)
        strategy = self.speaker_state.get("strategy", "sequential")

        # Track which agents have spoken to ensure all get a chance
        spoken_agents = set()
        last_response = ""
        max_loops = (
            len(mentioned_agents) * 3
        )  # Allow more loops for parallel
        iteration = 0

        while iteration < max_loops and len(spoken_agents) < len(
            mentioned_agents
        ):
            # Determine next speaker(s) using dynamic function
            # Avoid passing duplicate 'strategy' if it's present in speaker_state
            speaker_state = {
                k: v
                for k, v in (self.speaker_state or {}).items()
                if k != "strategy"
            }
            next_speakers = self.speaker_function(
                mentioned_agents,
                last_response,
                strategy=strategy,
                **speaker_state,
            )

            # Handle both single agent and multiple agents
            if isinstance(next_speakers, str):
                next_speakers = [next_speakers]

            # Filter out invalid agents
            valid_next_speakers = [
                agent
                for agent in next_speakers
                if agent in mentioned_agents
            ]

            if not valid_next_speakers:
                # If no valid mentions found, randomly select from unspoken agents
                unspoken_agents = [
                    agent
                    for agent in mentioned_agents
                    if agent not in spoken_agents
                ]
                if unspoken_agents:
                    valid_next_speakers = [
                        random.choice(unspoken_agents)
                    ]
                else:
                    # All agents have spoken, break the loop
                    break

            # Process agents based on strategy
            if strategy == "sequential":
                self._process_sequential_speakers(
                    valid_next_speakers, spoken_agents, img, imgs
                )
            elif strategy == "parallel":
                self._process_parallel_speakers(
                    valid_next_speakers, spoken_agents, img, imgs
                )

            iteration += 1

    def _process_sequential_speakers(
        self,
        speakers: List[str],
        spoken_agents: set,
        img: Optional[str],
        imgs: Optional[List[str]],
    ) -> None:
        """
        Process speakers sequentially.

        Args:
            speakers (List[str]): List of agent names to process in order.
            spoken_agents (set): Set of agent names that have already spoken.
            img (Optional[str]): Optional image input for the agents.
            imgs (Optional[List[str]]): Optional list of images for the agents.

        Returns:
            None
        """
        for next_speaker in speakers:
            if next_speaker in spoken_agents:
                continue  # Skip if already spoken

            response = self._get_agent_response(
                next_speaker, img, imgs
            )
            if response:
                spoken_agents.add(next_speaker)
                break  # Only process one agent in sequential mode

    def _process_parallel_speakers(
        self,
        speakers: List[str],
        spoken_agents: set,
        img: Optional[str],
        imgs: Optional[List[str]],
    ) -> None:
        """
        Process speakers in parallel.

        Args:
            speakers (List[str]): List of agent names to process in parallel.
            spoken_agents (set): Set of agent names that have already spoken.
            img (Optional[str]): Optional image input for the agents.
            imgs (Optional[List[str]]): Optional list of images for the agents.

        Returns:
            None
        """
        import concurrent.futures

        # Get responses from all valid agents
        responses = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_agent = {
                executor.submit(
                    self._get_agent_response, agent, img, imgs
                ): agent
                for agent in speakers
                if agent not in spoken_agents
            }

            for future in concurrent.futures.as_completed(
                future_to_agent
            ):
                agent = future_to_agent[future]
                try:
                    response = future.result()
                    if response:
                        responses.append(response)
                        spoken_agents.add(agent)
                except Exception as e:
                    logger.error(
                        f"Error getting response from {agent}: {e}"
                    )

    def _process_static_speakers(
        self,
        mentioned_agents: List[str],
        img: Optional[str],
        imgs: Optional[List[str]],
    ) -> None:
        """
        Process responses using a static speaker function.

        Args:
            mentioned_agents (List[str]): List of agent names to process.
            img (Optional[str]): Optional image input for the agents.
            imgs (Optional[List[str]]): Optional list of images for the agents.

        Returns:
            None
        """
        speaking_order = self._get_speaking_order(mentioned_agents)
        logger.info(f"Speaking order determined: {speaking_order}")

        # Get responses from mentioned agents in the determined order
        for agent_name in speaking_order:
            self._get_agent_response(agent_name, img, imgs)

    def _process_random_speaker(
        self,
        mentioned_agents: List[str],
        img: Optional[str],
        imgs: Optional[List[str]],
    ) -> None:
        """
        Process responses using the random speaker function.
        This function randomly selects a single agent from the mentioned agents
        to respond to the user query.
        """
        # Filter out invalid agents
        valid_agents = [
            name
            for name in mentioned_agents
            if name in self.agent_map
        ]

        if not valid_agents:
            raise AgentNotFoundError(
                "No valid agents found in the conversation"
            )

        # Randomly select exactly one agent to respond
        random_agent = random.choice(valid_agents)
        logger.info(f"Random speaker selected: {random_agent}")

        # Get response from the randomly selected agent
        self._get_agent_response(random_agent, img, imgs)

    def run(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
    ) -> str:
        """
        Process a task and get responses from agents. If no agents are mentioned,
        randomly selects agents to participate.

        Args:
            task (str): The user input or task to process.
            img (Optional[str]): Optional image input for the agents.
            imgs (Optional[List[str]]): Optional list of images for the agents.

        Returns:
            str: The formatted conversation history (format depends on output_type).

        Raises:
            GroupChatError: If an unexpected error occurs.
        """
        try:
            # Extract mentioned agents (or all agents if none mentioned)
            if "@" in task:
                mentioned_agents = self._extract_mentions(task)
            else:
                mentioned_agents = list(self.agent_map.keys())

            # Add user task to conversation
            self.conversation.add(role="User", content=task)

            # Process responses based on speaker function type
            if self.speaker_function == random_dynamic_speaker:
                self._process_dynamic_speakers(
                    mentioned_agents, img, imgs
                )
            elif self.speaker_function == random_speaker:
                # Use the specialized function for random_speaker
                self._process_random_speaker(
                    mentioned_agents, img, imgs
                )
            else:
                self._process_static_speakers(
                    mentioned_agents, img, imgs
                )

            return history_output_formatter(
                self.conversation, self.output_type
            )

        except Exception as e:
            logger.error(
                f"GroupChat: Unexpected error: {e} Traceback: {traceback.format_exc()}"
            )
            raise GroupChatError(
                f"GroupChat: Unexpected error occurred: {str(e)} Traceback: {traceback.format_exc()}"
            )

    def __call__(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
    ):
        return self.run(task=task, img=img, imgs=imgs)

    def _get_agent_response(
        self,
        agent_name: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Get response from a specific agent.

        Args:
            agent_name (str): Name of the agent to get response from.
            img (Optional[str]): Optional image for the task.
            imgs (Optional[List[str]]): Optional list of images for the task.

        Returns:
            Optional[str]: The agent's response or None if error.

        Raises:
            AgentNotFoundError: If the agent is not found.
        """
        agent = self.agent_map.get(agent_name)
        if not agent:
            raise AgentNotFoundError(
                f"Agent '{agent_name}' not found"
            )

        try:
            # Get the complete conversation history
            context = self.conversation.return_history_as_string()

            # Get response from agent
            if isinstance(agent, Agent):
                collaborative_task = get_collaborative_task_prompt(
                    context, agent_name
                )

                response = agent.run(
                    task=collaborative_task,
                    img=img,
                    imgs=imgs,
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
                return response

        except Exception as e:
            logger.error(
                f"Error getting response from {agent_name}: {e}"
            )
            self.conversation.add(
                role=agent_name,
                content=f"Error: Unable to generate response - {str(e)}",
            )
            return f"Error: Unable to generate response - {str(e)}"

        return None

    # Legacy methods for backward compatibility
    def reliability_check(self):
        """
        Validates the group chat configuration (legacy method).

        Raises:
            ValueError: If any required components are missing or invalid
        """
        self._validate_initialization()

        # Add legacy prompt augmentation if using old speaker functions
        if hasattr(self, "speaker_fn") and self.speaker_fn:
            for agent in self.agents:
                if isinstance(agent, Agent):
                    agent.system_prompt += (
                        MULTI_AGENT_COLLAB_PROMPT_TWO
                    )

    # Legacy run method for backward compatibility
    def run_legacy(
        self, task: str, img: str = None, *args, **kwargs
    ) -> str:
        """
        Legacy run method for backward compatibility with old GroupChat interface.
        """
        if not task or not isinstance(task, str):
            raise ValueError("Task must be a non-empty string")

        # Initialize conversation with context
        self.conversation.add(role="User", content=task)

        try:
            turn = 0
            # Determine a random number of conversation turns
            target_turns = random.randint(1, 4)
            logger.debug(
                f"Planning for approximately {target_turns} conversation turns"
            )

            # Keep track of which agent spoke last to create realistic exchanges
            last_speaker = None

            while turn < target_turns:

                # Select an agent to speak (different from the last speaker if possible)
                available_agents = self.agents.copy()

                if last_speaker and len(available_agents) > 1:
                    available_agents.remove(last_speaker)

                current_speaker = random.choice(available_agents)

                try:
                    # Build complete context with conversation history
                    conversation_history = (
                        self.conversation.return_history_as_string()
                    )

                    # Prepare a prompt that explicitly encourages responding to others
                    if last_speaker:
                        prompt = f"The previous message was from {last_speaker.agent_name}. As {current_speaker.agent_name}, please respond to what they and others have said about: {task}"
                    else:
                        prompt = f"As {current_speaker.agent_name}, please start the discussion about: {task}"

                    # Get the agent's response with full context awareness
                    message = current_speaker.run(
                        task=f"{conversation_history} {prompt}",
                    )

                    # Only add meaningful responses
                    if message and not message.isspace():
                        self.conversation.add(
                            role=current_speaker.agent_name,
                            content=message,
                        )

                        logger.info(
                            f"Turn {turn}, {current_speaker.agent_name} responded"
                        )

                        # Update the last speaker
                        last_speaker = current_speaker
                        turn += 1

                        # Occasionally end early to create natural variation
                        if (
                            turn > 3 and random.random() < 0.15
                        ):  # 15% chance to end after at least 3 turns
                            logger.debug(
                                "Random early conversation end"
                            )
                            break

                except Exception as e:
                    logger.error(
                        f"Error from {current_speaker.agent_name}: {e}"
                    )
                    # Skip this agent and continue conversation
                    continue

            return history_output_formatter(
                self.conversation, self.output_type
            )

        except Exception as e:
            logger.error(f"Error in chat: {e}")
            raise

    def batched_run(
        self, tasks: List[str], *args, **kwargs
    ) -> List[str]:
        """
        Runs multiple tasks in sequence.

        Args:
            tasks (List[str]): List of tasks to process
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            List[str]: List of conversation histories for each task

        Raises:
            ValueError: If tasks list is empty or invalid
        """
        if not tasks or not isinstance(tasks, list):
            raise ValueError(
                "Tasks must be a non-empty list of strings"
            )
        return [self.run(task, *args, **kwargs) for task in tasks]

    def concurrent_run(
        self, tasks: List[str], *args, **kwargs
    ) -> List[str]:
        """
        Runs multiple tasks concurrently using threads.

        Args:
            tasks (List[str]): List of tasks to process
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            List[str]: List of conversation histories for each task

        Raises:
            ValueError: If tasks list is empty or invalid
            RuntimeError: If concurrent execution fails
        """
        if not tasks or not isinstance(tasks, list):
            raise ValueError(
                "Tasks must be a non-empty list of strings"
            )

        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                return list(
                    executor.map(
                        lambda task: self.run(task, *args, **kwargs),
                        tasks,
                    )
                )
        except Exception as e:
            logger.error(f"Error in concurrent execution: {e}")
            raise RuntimeError(
                f"Concurrent execution failed: {str(e)}"
            )

    def _validate_speaker_function(self):
        """
        Validates the speaker function.
        """
        # This method is called to validate speaker functions
        pass
