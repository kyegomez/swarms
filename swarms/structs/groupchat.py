from typing import List, Dict, Optional, Union, Callable, Any
from pydantic import BaseModel, Field
from datetime import datetime
import json
from uuid import uuid4
import logging
from swarms.structs.agent import Agent
from swarms.structs.agents_available import showcase_available_agents

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Message(BaseModel):
    """Single message in the conversation"""

    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentMetadata(BaseModel):
    """Metadata for tracking agent state and configuration"""

    agent_name: str
    agent_type: str
    system_prompt: Optional[str] = None
    description: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)


class InteractionLog(BaseModel):
    """Log entry for a single interaction"""

    id: str = Field(default_factory=lambda: uuid4().hex)
    agent_name: str
    position: int
    input_text: str
    output_text: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GroupChatState(BaseModel):
    """Complete state of the group chat"""

    id: str = Field(default_factory=lambda: uuid4().hex)
    name: Optional[str] = None
    description: Optional[str] = None
    admin_name: str
    group_objective: str
    max_rounds: int
    rules: Optional[str] = None
    agent_metadata: List[AgentMetadata]
    messages: List[Message]
    interactions: List[InteractionLog]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class AgentWrapper:
    """Wrapper class to standardize agent interfaces"""

    def __init__(
        self,
        agent: Union["Agent", Callable],
        agent_name: str,
        system_prompt: Optional[str] = None,
    ):
        self.agent = agent
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self._validate_agent()

    def _validate_agent(self):
        """Validate that the agent has the required interface"""
        if hasattr(self.agent, "run"):
            self.run = self.agent.run
        elif callable(self.agent):
            self.run = self.agent
        else:
            raise ValueError(
                "Agent must either have a 'run' method or be callable"
            )

    def get_metadata(self) -> AgentMetadata:
        """Extract metadata from the agent"""
        return AgentMetadata(
            agent_name=self.agent_name,
            agent_type=type(self.agent).__name__,
            system_prompt=self.system_prompt,
            config={
                k: v
                for k, v in self.agent.__dict__.items()
                if isinstance(v, (str, int, float, bool, dict, list))
            },
        )


class GroupChat:
    """Enhanced GroupChat manager with state persistence and comprehensive logging.

    This class implements a multi-agent chat system with the following key features:
    - State persistence to disk
    - Comprehensive interaction logging
    - Configurable agent selection
    - Early stopping conditions
    - Conversation export capabilities

    The GroupChat coordinates multiple agents to have a goal-directed conversation,
    with one agent speaking at a time based on a selector agent's decisions.

    Attributes:
        name (Optional[str]): Name of the group chat
        description (Optional[str]): Description of the group chat's purpose
        agents (List[Union["Agent", Callable]]): List of participating agents
        max_rounds (int): Maximum number of conversation rounds
        admin_name (str): Name of the administrator
        group_objective (str): The goal/objective of the conversation
        selector_agent (Union["Agent", Callable]): Agent that selects next speaker
        rules (Optional[str]): Rules governing the conversation
        state_path (Optional[str]): Path to save conversation state
        showcase_agents_on (bool): Whether to showcase agent capabilities
    """

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        agents: List[Union["Agent", Callable]] = None,
        max_rounds: int = 10,
        admin_name: str = "Admin",
        group_objective: str = None,
        selector_agent: Union["Agent", Callable] = None,
        rules: Optional[str] = None,
        state_path: Optional[str] = None,
        showcase_agents_on: bool = False,
    ):
        """Initialize a new GroupChat instance.

        Args:
            name: Name of the group chat
            description: Description of the group chat's purpose
            agents: List of participating agents
            max_rounds: Maximum number of conversation rounds
            admin_name: Name of the administrator
            group_objective: The goal/objective of the conversation
            selector_agent: Agent that selects next speaker
            rules: Rules governing the conversation
            state_path: Path to save conversation state
            showcase_agents_on: Whether to showcase agent capabilities

        Raises:
            ValueError: If no agents are provided
        """
        self.name = name
        self.description = description
        self.agents = agents
        self.max_rounds = max_rounds
        self.admin_name = admin_name
        self.group_objective = group_objective
        self.selector_agent = selector_agent
        self.rules = rules
        self.state_path = state_path
        self.showcase_agents_on = showcase_agents_on

        if not agents:
            raise ValueError("At least two agents are required")

        # Generate unique state path if not provided
        self.state_path = (
            state_path or f"group_chat_{uuid4().hex}.json"
        )

        # Wrap all agents to standardize interface
        self.wrapped_agents = [
            AgentWrapper(
                agent,
                (
                    f"Agent_{i}"
                    if not hasattr(agent, "agent_name")
                    else agent.agent_name
                ),
            )
            for i, agent in enumerate(agents)
        ]

        # Configure selector agent
        self.selector_agent = AgentWrapper(
            selector_agent or self.wrapped_agents[0].agent,
            "Selector",
            "Select the next speaker based on the conversation context",
        )

        # Initialize conversation state
        self.state = GroupChatState(
            name=name,
            description=description,
            admin_name=admin_name,
            group_objective=group_objective,
            max_rounds=max_rounds,
            rules=rules,
            agent_metadata=[
                agent.get_metadata() for agent in self.wrapped_agents
            ],
            messages=[],
            interactions=[],
        )

        # Showcase agents if enabled
        if self.showcase_agents_on is True:
            self.showcase_agents()

    def showcase_agents(self):
        """Showcase available agents and update their system prompts.

        This method displays agent capabilities and updates each agent's
        system prompt with information about other agents in the group.
        """
        out = showcase_available_agents(
            name=self.name,
            description=self.description,
            agents=self.wrapped_agents,
        )

        for agent in self.wrapped_agents:
            # Initialize system_prompt if None
            if agent.system_prompt is None:
                agent.system_prompt = ""
            agent.system_prompt += out

    def save_state(self) -> None:
        """Save current conversation state to disk.

        The state is saved as a JSON file at the configured state_path.
        """
        with open(self.state_path, "w") as f:
            json.dump(self.state.dict(), f, default=str, indent=2)
        logger.info(f"State saved to {self.state_path}")

    @classmethod
    def load_state(cls, state_path: str) -> "GroupChat":
        """Load GroupChat from saved state.

        Args:
            state_path: Path to the saved state JSON file

        Returns:
            GroupChat: A new GroupChat instance with restored state

        Raises:
            FileNotFoundError: If state file doesn't exist
            json.JSONDecodeError: If state file is invalid JSON
        """
        with open(state_path, "r") as f:
            state_dict = json.load(f)

        # Convert loaded data back to state model
        state = GroupChatState(**state_dict)

        # Initialize with minimal config, then restore state
        instance = cls(
            name=state.name,
            admin_name=state.admin_name,
            agents=[],  # Temporary empty list
            group_objective=state.group_objective,
        )
        instance.state = state
        return instance

    def _log_interaction(
        self,
        agent_name: str,
        position: int,
        input_text: str,
        output_text: str,
    ) -> None:
        """Log a single interaction in the conversation.

        Args:
            agent_name: Name of the speaking agent
            position: Position in conversation sequence
            input_text: Input context provided to agent
            output_text: Agent's response
        """
        log_entry = InteractionLog(
            agent_name=agent_name,
            position=position,
            input_text=input_text,
            output_text=output_text,
            metadata={
                "current_agents": [
                    a.agent_name for a in self.wrapped_agents
                ],
                "round": position // len(self.wrapped_agents),
            },
        )
        self.state.interactions.append(log_entry)
        self.save_state()

    def _add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history.

        Args:
            role: Speaker's role/name
            content: Message content
        """
        message = Message(role=role, content=content)
        self.state.messages.append(message)
        self.save_state()

    def select_next_speaker(
        self, last_speaker: AgentWrapper
    ) -> AgentWrapper:
        """Select the next speaker using the selector agent.

        Args:
            last_speaker: The agent who spoke last

        Returns:
            AgentWrapper: The next agent to speak

        Note:
            Falls back to round-robin selection if selector agent fails
        """
        conversation_history = "\n".join(
            [
                f"{msg.role}: {msg.content}"
                for msg in self.state.messages
            ]
        )

        selection_prompt = f"""
        Current speakers: {[agent.agent_name for agent in self.wrapped_agents]}
        Last speaker: {last_speaker.agent_name}
        Group objective: {self.state.group_objective}
        
        Based on the conversation history and group objective, select the next most appropriate speaker.
        Only return the speaker's name.
        
        Conversation history:
        {conversation_history}
        """

        try:
            next_speaker_name = self.selector_agent.run(
                selection_prompt
            ).strip()
            return next(
                agent
                for agent in self.wrapped_agents
                if agent.agent_name in next_speaker_name
            )
        except (StopIteration, Exception) as e:
            logger.warning(
                f"Selector agent failed: {str(e)}. Falling back to round-robin."
            )
            # Fallback to round-robin if selection fails
            current_idx = self.wrapped_agents.index(last_speaker)
            return self.wrapped_agents[
                (current_idx + 1) % len(self.wrapped_agents)
            ]

    def run(self, task: str) -> str:
        """Execute the group chat conversation.

        Args:
            task: The initial task/question to discuss

        Returns:
            str: The final response from the conversation

        Raises:
            Exception: If any error occurs during execution
        """
        try:
            logger.info(f"Starting GroupChat with task: {task}")
            self._add_message(self.state.admin_name, task)

            current_speaker = self.wrapped_agents[0]
            final_response = None

            for round_num in range(self.state.max_rounds):
                # Select next speaker
                current_speaker = self.select_next_speaker(
                    current_speaker
                )
                logger.info(
                    f"Selected speaker: {current_speaker.agent_name}"
                )

                # Prepare context and get response
                conversation_history = "\n".join(
                    [
                        f"{msg.role}: {msg.content}"
                        for msg in self.state.messages[
                            -10:
                        ]  # Last 10 messages for context
                    ]
                )

                try:
                    response = current_speaker.run(
                        conversation_history
                    )
                    final_response = response
                except Exception as e:
                    logger.error(
                        f"Agent {current_speaker.agent_name} failed: {str(e)}"
                    )
                    continue

                # Log interaction and add to message history
                self._log_interaction(
                    current_speaker.agent_name,
                    round_num,
                    conversation_history,
                    response,
                )
                self._add_message(
                    current_speaker.agent_name, response
                )

                # Optional: Add early stopping condition based on response content
                if (
                    "TASK_COMPLETE" in response
                    or "CONCLUSION" in response
                ):
                    logger.info(
                        "Task completion detected, ending conversation"
                    )
                    break

            return final_response or "No valid response generated"

        except Exception as e:
            logger.error(f"Error in GroupChat execution: {str(e)}")
            raise

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Return a summary of the conversation.

        Returns:
            Dict containing conversation metrics and status
        """
        return {
            "id": self.state.id,
            "total_interactions": len(self.state.interactions),
            "participating_agents": [
                agent.agent_name for agent in self.wrapped_agents
            ],
            "conversation_length": len(self.state.messages),
            "duration": (
                datetime.utcnow() - self.state.created_at
            ).total_seconds(),
            "objective_completed": any(
                "TASK_COMPLETE" in msg.content
                for msg in self.state.messages
            ),
        }

    def export_conversation(
        self, format: str = "json"
    ) -> Union[str, Dict]:
        """Export the conversation in the specified format.

        Args:
            format: Output format ("json" or "text")

        Returns:
            Union[str, Dict]: Conversation in requested format

        Raises:
            ValueError: If format is not supported
        """
        if format == "json":
            return self.state.dict()
        elif format == "text":
            return "\n".join(
                [
                    f"{msg.role} ({msg.timestamp}): {msg.content}"
                    for msg in self.state.messages
                ]
            )
        else:
            raise ValueError(f"Unsupported export format: {format}")
