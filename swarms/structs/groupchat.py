import concurrent.futures
import random
from datetime import datetime
from typing import Callable, List

from loguru import logger
from pydantic import BaseModel, Field

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.multi_agent_exec import get_agents_info
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.prompts.multi_agent_collab_prompt import (
    MULTI_AGENT_COLLAB_PROMPT_TWO,
)


class AgentResponse(BaseModel):
    agent_name: str
    role: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    turn_number: int
    preceding_context: List[str] = Field(default_factory=list)


SpeakerFunction = Callable[[List[str], "Agent"], bool]


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
    A class that manages conversations between multiple AI agents.

    This class facilitates group chat interactions between multiple agents, where agents
    can communicate with each other based on a specified speaker function. It handles
    conversation flow, message history, and agent coordination.

    Attributes:
        name (str): Name of the group chat
        description (str): Description of the group chat's purpose
        agents (List[Agent]): List of Agent instances participating in the chat
        speaker_fn (SpeakerFunction): Function determining which agents can speak
        max_loops (int): Maximum number of conversation turns
        conversation (Conversation): Stores the chat history

    Args:
        name (str, optional): Name of the group chat. Defaults to "GroupChat".
        description (str, optional): Description of the chat. Defaults to "A group chat for multiple agents".
        agents (List[Agent], optional): List of participating agents. Defaults to empty list.
        speaker_fn (SpeakerFunction, optional): Speaker selection function. Defaults to round_robin.
        max_loops (int, optional): Maximum conversation turns. Defaults to 1.

    Raises:
        ValueError: If invalid initialization parameters are provided
    """

    def __init__(
        self,
        name: str = "GroupChat",
        description: str = "A group chat for multiple agents",
        agents: List[Agent] = [],
        speaker_fn: SpeakerFunction = round_robin,
        max_loops: int = 1,
        rules: str = "",
        output_type: str = "string",
    ):
        self.name = name
        self.description = description
        self.agents = agents
        self.speaker_fn = speaker_fn
        self.max_loops = max_loops
        self.output_type = output_type
        self.rules = rules

        self.conversation = Conversation(
            time_enabled=False, rules=rules
        )

        agent_context = f"\n Group Chat Name: {self.name}\nGroup Chat Description: {self.description}\n Agents in your Group Chat: {get_agents_info(self.agents)}"
        self.conversation.add(role="System", content=agent_context)

        self.reliability_check()

    def reliability_check(self):
        """
        Validates the group chat configuration.

        Raises:
            ValueError: If any required components are missing or invalid
        """

        if len(self.agents) < 2:
            raise ValueError(
                "At least two agents are required for a group chat"
            )

        if self.max_loops <= 0:
            raise ValueError("Max loops must be greater than 0")

        for agent in self.agents:
            agent.system_prompt += MULTI_AGENT_COLLAB_PROMPT_TWO

    def run(self, task: str, img: str = None, *args, **kwargs) -> str:
        """
        Executes a dynamic conversation between agents about the given task.

        Agents are selected randomly to speak, creating a more natural flow
        with varying conversation lengths.

        Args:
            task (str): The task or topic for agents to discuss
            img (str, optional): Image input for the conversation. Defaults to None.
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            str: Complete conversation history as a string

        Raises:
            ValueError: If task is empty or invalid
            Exception: If any error occurs during conversation
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

#     agents = [agent1, agent2]

#     chat = GroupChat(
#         name="Investment Advisory",
#         description="Financial and tax analysis group",
#         agents=agents,
#         speaker_fn=expertise_based,
#     )

#     history = chat.run(
#         "How to optimize tax strategy for investments?"
#     )
#     print(history.model_dump_json(indent=2))
