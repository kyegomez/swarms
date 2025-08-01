import math
import random
import time
import threading
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from loguru import logger

from swarms import Agent


@dataclass
class Position:
    """Represents a 2D position on the map."""

    x: float
    y: float

    def distance_to(self, other: "Position") -> float:
        """Calculate Euclidean distance to another position."""
        return math.sqrt(
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2
        )


@dataclass
class AgentState:
    """Represents the state of an agent in the simulation."""

    agent: Agent
    position: Position
    target_position: Optional[Position] = None
    is_in_conversation: bool = False
    conversation_partner: Optional[str] = (
        None  # Kept for backwards compatibility
    )
    conversation_partners: List[str] = (
        None  # NEW: Support multiple conversation partners
    )
    conversation_id: Optional[str] = (
        None  # NEW: Track which conversation group the agent is in
    )
    conversation_thread: Optional[threading.Thread] = None
    conversation_history: List[str] = None
    movement_speed: float = 1.0
    conversation_radius: float = 3.0

    def __post_init__(self):
        """Initialize conversation history and partners list if not provided."""
        if self.conversation_history is None:
            self.conversation_history = []
        if self.conversation_partners is None:
            self.conversation_partners = []


class ConversationManager:
    """Manages active conversations between agents, supporting both 1-on-1 and group conversations."""

    def __init__(self):
        """Initialize the conversation manager."""
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        self.conversation_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.group_join_threshold = 15.0  # Distance within which agents can join existing conversations

    def start_conversation(
        self,
        agent1: AgentState,
        agent2: AgentState,
        topic: str = None,
    ) -> str:
        """
        Start a conversation between two agents.

        Args:
            agent1: First agent state
            agent2: Second agent state
            topic: Conversation topic (if not provided, will use default topics)

        Returns:
            Conversation ID
        """
        conversation_id = f"conv_{agent1.agent.agent_name}_{agent2.agent.agent_name}_{int(time.time())}"

        if topic is None:
            # Default topics if none specified
            topics = [
                "What are the most promising investment opportunities in the current market?",
                "How should risk management strategies adapt to market volatility?",
                "What role does artificial intelligence play in modern trading?",
                "How do geopolitical events impact global financial markets?",
                "What are the key indicators for identifying market trends?",
            ]
            topic = random.choice(topics)

        with self.conversation_lock:
            self.active_conversations[conversation_id] = {
                "participants": [
                    agent1,
                    agent2,
                ],  # NEW: Store list of participants instead of agent1/agent2
                "topic": topic,
                "start_time": time.time(),
                "status": "starting",
                "conversation_center": self._calculate_conversation_center(
                    [agent1, agent2]
                ),  # NEW: Track conversation location
                "max_participants": 6,  # NEW: Limit group size
            }

            # Mark agents as in conversation
            self._add_agent_to_conversation(
                agent1, conversation_id, [agent2]
            )
            self._add_agent_to_conversation(
                agent2, conversation_id, [agent1]
            )

        # Start conversation in thread
        self.executor.submit(self._run_conversation, conversation_id)

        return conversation_id

    def try_join_conversation(
        self, agent: AgentState, conversation_id: str
    ) -> bool:
        """
        Try to add an agent to an existing conversation.

        Args:
            agent: Agent that wants to join
            conversation_id: ID of the conversation to join

        Returns:
            True if agent was successfully added, False otherwise
        """
        with self.conversation_lock:
            if conversation_id not in self.active_conversations:
                return False

            conversation = self.active_conversations[conversation_id]

            # Check if conversation is active and not at max capacity
            if (
                conversation["status"] != "active"
                or len(conversation["participants"])
                >= conversation["max_participants"]
                or agent.is_in_conversation
            ):
                return False

            # Add agent to conversation
            conversation["participants"].append(agent)
            other_participants = [
                p for p in conversation["participants"] if p != agent
            ]
            self._add_agent_to_conversation(
                agent, conversation_id, other_participants
            )

            # Update conversation center
            conversation["conversation_center"] = (
                self._calculate_conversation_center(
                    conversation["participants"]
                )
            )

            logger.info(
                f"👥 {agent.agent.agent_name} joined conversation (now {len(conversation['participants'])} participants)"
            )
            return True

    def _add_agent_to_conversation(
        self,
        agent: AgentState,
        conversation_id: str,
        other_participants: List[AgentState],
    ):
        """
        Helper method to mark an agent as being in a conversation.

        Args:
            agent: Agent to add
            conversation_id: ID of the conversation
            other_participants: Other agents in the conversation
        """
        agent.is_in_conversation = True
        agent.conversation_id = conversation_id
        agent.conversation_partners = [
            p.agent.agent_name for p in other_participants
        ]
        # Keep backwards compatibility
        agent.conversation_partner = (
            other_participants[0].agent.agent_name
            if other_participants
            else None
        )

    def _calculate_conversation_center(
        self, participants: List[AgentState]
    ) -> Position:
        """
        Calculate the center point of a conversation group.

        Args:
            participants: List of agents in the conversation

        Returns:
            Position representing the center of the conversation
        """
        if not participants:
            return Position(0, 0)

        avg_x = sum(p.position.x for p in participants) / len(
            participants
        )
        avg_y = sum(p.position.y for p in participants) / len(
            participants
        )
        return Position(avg_x, avg_y)

    def _run_conversation(self, conversation_id: str):
        """
        Run a conversation between multiple agents (group conversation support).

        Args:
            conversation_id: ID of the conversation to run
        """
        try:
            conversation_data = self.active_conversations[
                conversation_id
            ]
            participants = conversation_data["participants"]
            topic = conversation_data["topic"]

            # Update status
            conversation_data["status"] = "active"

            participant_names = [
                p.agent.agent_name for p in participants
            ]
            logger.info(
                f"🗣️  Group conversation started: {', '.join(participant_names)}"
            )
            logger.info(f"📝 Topic: {topic}")

            # Random conversation depth between 2-8 loops for groups
            conversation_loops = random.randint(2, 8)
            logger.debug(
                f"💬 Conversation depth: {conversation_loops} exchanges"
            )

            # Start the conversation with introductions
            conversation_history = []

            # Each participant introduces themselves
            for participant in participants:
                other_names = [
                    p.agent.agent_name
                    for p in participants
                    if p != participant
                ]
                if len(other_names) == 1:
                    intro = f"Hi! I'm {participant.agent.agent_name} - {participant.agent.agent_description}. Nice to meet you, {other_names[0]}!"
                else:
                    intro = f"Hello everyone! I'm {participant.agent.agent_name} - {participant.agent.agent_description}. Great to meet you all!"

                conversation_history.append(
                    f"{participant.agent.agent_name}: {intro}"
                )

            # Continue with the topic discussion
            current_message = (
                f"So, what do you all think about this: {topic}"
            )

            # Rotate speakers for more dynamic conversation
            speaker_index = 0

            for i in range(conversation_loops):
                current_speaker = participants[
                    speaker_index % len(participants)
                ]
                other_participants = [
                    p for p in participants if p != current_speaker
                ]
                other_names = [
                    p.agent.agent_name for p in other_participants
                ]

                # Check if new agents joined during conversation
                if len(participants) != len(
                    conversation_data["participants"]
                ):
                    participants = conversation_data["participants"]
                    logger.info(
                        f"👥 Updated participant list: {[p.agent.agent_name for p in participants]}"
                    )

                # Build context for group conversation
                full_context = f"""Group conversation in progress:
{chr(10).join(conversation_history[-6:])}  # Show last 6 exchanges for context

{current_speaker.agent.agent_name}, respond to the ongoing discussion: {current_message}

You're talking with: {', '.join(other_names)}
Participants: {', '.join([f"{p.agent.agent_name} ({p.agent.agent_description})" for p in other_participants])}

Keep it SHORT and conversational! Engage with the group naturally."""

                # Get response from current speaker
                try:
                    response = current_speaker.agent.run(
                        task=full_context
                    )

                    # Clean up the response
                    if response:
                        response = response.strip()
                        if response.startswith(
                            f"{current_speaker.agent.agent_name}:"
                        ):
                            response = response.replace(
                                f"{current_speaker.agent.agent_name}:",
                                "",
                            ).strip()
                    else:
                        response = f"[No response from {current_speaker.agent.agent_name}]"
                        logger.warning(
                            f"⚠️  No response from agent {current_speaker.agent.agent_name}"
                        )
                except Exception as e:
                    logger.exception(
                        f"❌ Error getting response from {current_speaker.agent.agent_name}: {str(e)}"
                    )
                    response = f"[Error getting response from {current_speaker.agent.agent_name}]"

                conversation_entry = (
                    f"{current_speaker.agent.agent_name}: {response}"
                )
                conversation_history.append(conversation_entry)

                # Update current message and rotate speaker
                current_message = response
                speaker_index += 1

            # Store conversation results
            conversation_data["history"] = conversation_history
            conversation_data["end_time"] = time.time()
            conversation_data["status"] = "completed"
            conversation_data["conversation_loops"] = (
                conversation_loops
            )

            # Update each participant's history
            for participant in participants:
                other_names = [
                    p.agent.agent_name
                    for p in participants
                    if p != participant
                ]
                participant.conversation_history.append(
                    {
                        "partners": other_names,  # NEW: Multiple partners
                        "partner": (
                            other_names[0] if other_names else "group"
                        ),  # Backwards compatibility
                        "topic": topic,
                        "timestamp": time.time(),
                        "history": conversation_history,
                        "loops": conversation_loops,
                        "group_size": len(
                            participants
                        ),  # NEW: Track group size
                    }
                )

            logger.success(
                f"✅ Group conversation completed: {', '.join(participant_names)} ({conversation_loops} exchanges)"
            )

        except Exception as e:
            logger.exception(
                f"❌ Error in conversation {conversation_id}: {str(e)}"
            )
            conversation_data["status"] = "error"
            conversation_data["error"] = str(e)

        finally:
            # Free up all participants
            with self.conversation_lock:
                if conversation_id in self.active_conversations:
                    participants = self.active_conversations[
                        conversation_id
                    ]["participants"]
                    for participant in participants:
                        participant.is_in_conversation = False
                        participant.conversation_partner = None
                        participant.conversation_partners = []
                        participant.conversation_id = None

    def get_active_conversations(self) -> Dict[str, Dict[str, Any]]:
        """Get all active conversations."""
        with self.conversation_lock:
            return {
                k: v
                for k, v in self.active_conversations.items()
                if v["status"] in ["starting", "active"]
            }


class AgentMapSimulation:
    """
    A simulation system where agents move on a 2D map and engage in conversations
    when they come into proximity with each other.
    """

    def __init__(
        self,
        map_width: float = 100.0,
        map_height: float = 100.0,
        proximity_threshold: float = 5.0,
        update_interval: float = 1.0,
    ):
        """
        Initialize the agent map simulation.

        Args:
            map_width: Width of the simulation map
            map_height: Height of the simulation map
            proximity_threshold: Distance threshold for triggering conversations
            update_interval: Time interval between simulation updates in seconds
        """
        self.map_width = map_width
        self.map_height = map_height
        self.proximity_threshold = proximity_threshold
        self.update_interval = update_interval

        self.agents: Dict[str, AgentState] = {}
        self.conversation_manager = ConversationManager()
        self.running = False
        self.simulation_thread: Optional[threading.Thread] = None

        # Task-specific settings
        self.current_task: Optional[str] = None
        self.task_mode: bool = False

        # Visualization
        self.fig = None
        self.ax = None
        self.agent_plots = {}
        self.conversation_lines = {}

    def add_agent(
        self,
        agent: Agent,
        position: Optional[Position] = None,
        movement_speed: float = 1.0,
        conversation_radius: float = 3.0,
    ) -> str:
        """
        Add an agent to the simulation.

        Args:
            agent: The Agent instance to add
            position: Starting position (random if not specified)
            movement_speed: Speed of agent movement
            conversation_radius: Radius for conversation detection

        Returns:
            Agent ID in the simulation
        """
        if position is None:
            position = Position(
                x=random.uniform(0, self.map_width),
                y=random.uniform(0, self.map_height),
            )

        agent_state = AgentState(
            agent=agent,
            position=position,
            movement_speed=movement_speed,
            conversation_radius=conversation_radius,
        )

        self.agents[agent.agent_name] = agent_state
        logger.info(
            f"➕ Added agent '{agent.agent_name}' at position ({position.x:.1f}, {position.y:.1f})"
        )

        return agent.agent_name

    def batch_add_agents(
        self,
        agents: List[Agent],
        positions: List[Position],
        movement_speeds: List[float],
        conversation_radii: List[float],
    ):
        """
        Add multiple agents to the simulation.

        Args:
            agents: List of Agent instances to add
            positions: List of starting positions for each agent
            movement_speeds: List of movement speeds for each agent
            conversation_radii: List of conversation radii for each agent
        """
        for i in range(len(agents)):
            self.add_agent(
                agents[i],
                positions[i],
                movement_speeds[i],
                conversation_radii[i],
            )

    def remove_agent(self, agent_name: str) -> bool:
        """
        Remove an agent from the simulation.

        Args:
            agent_name: Name of the agent to remove

        Returns:
            True if agent was removed, False if not found
        """
        if agent_name in self.agents:
            # End any active conversations
            agent_state = self.agents[agent_name]
            if agent_state.is_in_conversation:
                agent_state.is_in_conversation = False
                agent_state.conversation_partner = None

            del self.agents[agent_name]
            logger.info(
                f"➖ Removed agent '{agent_name}' from simulation"
            )
            return True
        return False

    def _generate_random_target(
        self, agent_state: AgentState
    ) -> Position:
        """
        Generate a random target position for an agent.

        Args:
            agent_state: The agent state

        Returns:
            Random target position
        """
        return Position(
            x=random.uniform(0, self.map_width),
            y=random.uniform(0, self.map_height),
        )

    def _move_agent(self, agent_state: AgentState, dt: float):
        """
        Move an agent towards its target position.

        Args:
            agent_state: The agent state to move
            dt: Time delta for movement calculation
        """
        if agent_state.is_in_conversation:
            return  # Don't move if in conversation

        # Generate new target if none exists or reached current target
        if (
            agent_state.target_position is None
            or agent_state.position.distance_to(
                agent_state.target_position
            )
            < 1.0
        ):
            agent_state.target_position = (
                self._generate_random_target(agent_state)
            )

        # Calculate movement direction
        dx = agent_state.target_position.x - agent_state.position.x
        dy = agent_state.target_position.y - agent_state.position.y
        distance = math.sqrt(dx * dx + dy * dy)

        if distance > 0:
            # Normalize direction and apply movement
            move_distance = agent_state.movement_speed * dt
            if move_distance >= distance:
                agent_state.position = agent_state.target_position
            else:
                agent_state.position.x += (
                    dx / distance
                ) * move_distance
                agent_state.position.y += (
                    dy / distance
                ) * move_distance

    def _check_proximity(self):
        """Check for agents in proximity and start conversations or join existing ones."""
        try:
            agent_list = list(self.agents.values())

            # First, check if any free agents can join existing conversations
            active_conversations = (
                self.conversation_manager.get_active_conversations()
            )
            for agent in agent_list:
                if agent.is_in_conversation:
                    continue

                # Check if agent is close to any active conversation
                for (
                    conv_id,
                    conv_data,
                ) in active_conversations.items():
                    if "conversation_center" in conv_data:
                        conversation_center = conv_data[
                            "conversation_center"
                        ]
                        distance_to_conversation = (
                            agent.position.distance_to(
                                conversation_center
                            )
                        )

                        # If agent is close to conversation center, try to join
                        if (
                            distance_to_conversation
                            <= self.conversation_manager.group_join_threshold
                        ):
                            if self.conversation_manager.try_join_conversation(
                                agent, conv_id
                            ):
                                break  # Agent joined, stop checking other conversations

            # Then, check for new conversations between free agents
            for i in range(len(agent_list)):
                for j in range(i + 1, len(agent_list)):
                    agent1 = agent_list[i]
                    agent2 = agent_list[j]

                    # Skip if either agent is already in conversation
                    if (
                        agent1.is_in_conversation
                        or agent2.is_in_conversation
                    ):
                        continue

                    distance = agent1.position.distance_to(
                        agent2.position
                    )

                    if distance <= self.proximity_threshold:
                        # Use the current task if in task mode, otherwise let conversation manager choose
                        topic = (
                            self.current_task
                            if self.task_mode
                            else None
                        )

                        # Start new conversation
                        self.conversation_manager.start_conversation(
                            agent1, agent2, topic
                        )
        except Exception as e:
            logger.exception(f"❌ Error checking proximity: {str(e)}")

    def _simulation_loop(self):
        """Main simulation loop."""
        try:
            last_time = time.time()

            while self.running:
                try:
                    current_time = time.time()
                    dt = current_time - last_time
                    last_time = current_time

                    # Move all agents
                    for agent_state in self.agents.values():
                        try:
                            self._move_agent(agent_state, dt)
                        except Exception as e:
                            logger.exception(
                                f"❌ Error moving agent {agent_state.agent.agent_name}: {str(e)}"
                            )

                    # Check for proximity-based conversations
                    self._check_proximity()

                    # Sleep for update interval
                    time.sleep(self.update_interval)
                except Exception as e:
                    logger.exception(
                        f"❌ Error in simulation loop: {str(e)}"
                    )
                    time.sleep(1)  # Brief pause before retry
        except Exception as e:
            logger.exception(
                f"❌ Critical error in simulation loop: {str(e)}"
            )
            self.running = False

    def start_simulation(self):
        """Start the simulation."""
        if self.running:
            logger.warning("⚠️  Simulation is already running")
            return

        try:
            self.running = True
            self.simulation_thread = threading.Thread(
                target=self._simulation_loop
            )
            self.simulation_thread.daemon = True
            self.simulation_thread.start()

            logger.success("🚀 Agent map simulation started")
        except Exception as e:
            logger.exception(
                f"❌ Failed to start simulation: {str(e)}"
            )
            self.running = False

    def stop_simulation(self):
        """Stop the simulation."""
        if not self.running:
            logger.warning("⚠️  Simulation is not running")
            return

        try:
            self.running = False
            if self.simulation_thread:
                self.simulation_thread.join(timeout=5.0)

            logger.success("🛑 Agent map simulation stopped")
        except Exception as e:
            logger.exception(
                f"❌ Error stopping simulation: {str(e)}"
            )

    def get_simulation_state(self) -> Dict[str, Any]:
        """
        Get current simulation state.

        Returns:
            Dictionary containing simulation state information
        """
        try:
            active_conversations = (
                self.conversation_manager.get_active_conversations()
            )

            agents_info = {}
            for name, state in self.agents.items():
                try:
                    agents_info[name] = {
                        "position": (
                            state.position.x,
                            state.position.y,
                        ),
                        "is_in_conversation": state.is_in_conversation,
                        "conversation_partner": getattr(
                            state, "conversation_partner", None
                        ),  # Backwards compatibility
                        "conversation_partners": getattr(
                            state, "conversation_partners", []
                        ),  # NEW: Multiple partners
                        "conversation_id": getattr(
                            state, "conversation_id", None
                        ),  # NEW: Current conversation ID
                        "conversation_count": len(
                            getattr(state, "conversation_history", [])
                        ),
                    }
                except Exception as e:
                    logger.exception(
                        f"❌ Error getting state for agent {name}: {str(e)}"
                    )
                    agents_info[name] = {
                        "position": (0, 0),
                        "is_in_conversation": False,
                        "conversation_partner": None,
                        "conversation_partners": [],
                        "conversation_id": None,
                        "conversation_count": 0,
                    }

            return {
                "agents": agents_info,
                "active_conversations": (
                    len(active_conversations)
                    if active_conversations
                    else 0
                ),
                "total_conversations": (
                    len(
                        self.conversation_manager.active_conversations
                    )
                    if self.conversation_manager.active_conversations
                    else 0
                ),
                "map_size": (self.map_width, self.map_height),
                "running": self.running,
            }
        except Exception as e:
            logger.exception(
                f"❌ Critical error getting simulation state: {str(e)}"
            )
            return {
                "agents": {},
                "active_conversations": 0,
                "total_conversations": 0,
                "map_size": (self.map_width, self.map_height),
                "running": self.running,
            }

    def print_status(self):
        """Print current simulation status."""
        try:
            state = self.get_simulation_state()

            logger.info("\n" + "=" * 60)
            logger.info("🗺️  AGENT MAP SIMULATION STATUS")
            logger.info("=" * 60)
            logger.info(
                f"Map Size: {state['map_size'][0]}x{state['map_size'][1]}"
            )
            logger.info(f"Running: {state['running']}")
            logger.info(
                f"Active Conversations: {state['active_conversations']}"
            )
            logger.info(
                f"Total Conversations: {state['total_conversations']}"
            )
            logger.info(f"Agents: {len(state['agents'])}")
            logger.info("\nAgent Positions:")

            for name, info in state["agents"].items():
                status = (
                    "🗣️  Talking"
                    if info["is_in_conversation"]
                    else "🚶 Moving"
                )

                # Enhanced partner display for group conversations
                if (
                    info["is_in_conversation"]
                    and info["conversation_partners"]
                ):
                    if len(info["conversation_partners"]) == 1:
                        partner = f" with {info['conversation_partners'][0]}"
                    else:
                        partner = f" in group with {', '.join(info['conversation_partners'][:2])}"
                        if len(info["conversation_partners"]) > 2:
                            partner += f" +{len(info['conversation_partners']) - 2} more"
                else:
                    partner = ""

                logger.info(
                    f"  {name}: ({info['position'][0]:.1f}, {info['position'][1]:.1f}) - {status}{partner} - {info['conversation_count']} conversations"
                )
            logger.info("=" * 60)
        except Exception as e:
            logger.exception(
                f"❌ Error printing simulation status: {str(e)}"
            )

    def setup_visualization(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Set up the matplotlib visualization.

        Args:
            figsize: Figure size for the plot
        """
        try:
            # Set backend to ensure compatibility
            import matplotlib

            matplotlib.use(
                "TkAgg"
            )  # Use TkAgg backend for better compatibility

            plt.ion()  # Turn on interactive mode
            self.fig, self.ax = plt.subplots(figsize=figsize)
            self.ax.set_xlim(0, self.map_width)
            self.ax.set_ylim(0, self.map_height)
            self.ax.set_aspect("equal")
            self.ax.grid(True, alpha=0.3)
            self.ax.set_title(
                "Agent Map Simulation", fontsize=16, fontweight="bold"
            )
            self.ax.set_xlabel("X Position")
            self.ax.set_ylabel("Y Position")

            # Create legend elements
            legend_elements = [
                patches.Circle(
                    (0, 0),
                    1,
                    color="blue",
                    alpha=0.7,
                    label="Available Agent",
                ),
                patches.Circle(
                    (0, 0),
                    1,
                    color="red",
                    alpha=0.7,
                    label="Agent in Conversation",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    color="purple",
                    linewidth=2,
                    alpha=0.7,
                    label="Conversation Link",
                ),
            ]
            self.ax.legend(handles=legend_elements, loc="upper right")

            # Show the window initially
            plt.show(block=False)
            plt.pause(0.1)  # Small pause to ensure window appears

            logger.success(
                "📊 Visualization setup complete - window should be visible now!"
            )

        except Exception as e:
            logger.exception(
                f"⚠️  Error setting up visualization: {str(e)}"
            )
            logger.warning("📊 Continuing without visualization...")
            self.fig = None
            self.ax = None

    def update_visualization(self):
        """Update the visualization with current agent positions and conversations."""
        if self.fig is None or self.ax is None:
            # Silently skip if visualization not available
            return

        try:
            # Clear previous plots
            for plot in self.agent_plots.values():
                if plot in self.ax.patches:
                    plot.remove()
            for line in self.conversation_lines.values():
                if line in self.ax.lines:
                    line.remove()

            # Clear text annotations
            for text in self.ax.texts[:]:
                if (
                    text not in self.ax.legend().get_texts()
                ):  # Keep legend text
                    text.remove()

            self.agent_plots.clear()
            self.conversation_lines.clear()

            # Plot agents
            for name, agent_state in self.agents.items():
                color = (
                    "red"
                    if agent_state.is_in_conversation
                    else "blue"
                )
                alpha = 0.8 if agent_state.is_in_conversation else 0.6

                # Draw agent circle
                circle = patches.Circle(
                    (agent_state.position.x, agent_state.position.y),
                    radius=1.5,
                    color=color,
                    alpha=alpha,
                    zorder=2,
                )
                self.ax.add_patch(circle)
                self.agent_plots[name] = circle

                # Add agent name label
                self.ax.text(
                    agent_state.position.x,
                    agent_state.position.y + 2.5,
                    name,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        alpha=0.8,
                    ),
                )

                # Draw conversation radius (only for non-conversing agents)
                if not agent_state.is_in_conversation:
                    radius_circle = patches.Circle(
                        (
                            agent_state.position.x,
                            agent_state.position.y,
                        ),
                        radius=self.proximity_threshold,
                        fill=False,
                        edgecolor=color,
                        alpha=0.2,
                        linestyle="--",
                        zorder=1,
                    )
                    self.ax.add_patch(radius_circle)

            # Draw conversation connections (supports both 1-on-1 and group conversations)
            active_conversations = (
                self.conversation_manager.get_active_conversations()
            )
            for conv_id, conv_data in active_conversations.items():
                participants = conv_data.get("participants", [])

                if len(participants) == 2:
                    # Traditional 1-on-1 conversation line
                    agent1, agent2 = participants
                    line = plt.Line2D(
                        [agent1.position.x, agent2.position.x],
                        [agent1.position.y, agent2.position.y],
                        color="purple",
                        linewidth=3,
                        alpha=0.7,
                        zorder=3,
                    )
                    self.ax.add_line(line)
                    self.conversation_lines[conv_id] = line

                    # Add conversation topic at midpoint
                    mid_x = (
                        agent1.position.x + agent2.position.x
                    ) / 2
                    mid_y = (
                        agent1.position.y + agent2.position.y
                    ) / 2

                elif len(participants) > 2:
                    # Group conversation - draw circle around conversation center
                    conversation_center = conv_data.get(
                        "conversation_center"
                    )
                    if conversation_center:
                        # Calculate group radius based on spread of participants
                        max_distance = max(
                            p.position.distance_to(
                                conversation_center
                            )
                            for p in participants
                        )
                        group_radius = max(
                            5.0, max_distance + 2.0
                        )  # At least 5 units radius

                        # Draw conversation circle
                        circle = patches.Circle(
                            (
                                conversation_center.x,
                                conversation_center.y,
                            ),
                            radius=group_radius,
                            fill=False,
                            edgecolor="purple",
                            linewidth=3,
                            alpha=0.6,
                            linestyle="-",
                            zorder=3,
                        )
                        self.ax.add_patch(circle)
                        self.conversation_lines[conv_id] = circle

                        # Draw lines connecting each participant to the center
                        for participant in participants:
                            line = plt.Line2D(
                                [
                                    participant.position.x,
                                    conversation_center.x,
                                ],
                                [
                                    participant.position.y,
                                    conversation_center.y,
                                ],
                                color="purple",
                                linewidth=1.5,
                                alpha=0.4,
                                zorder=2,
                            )
                            self.ax.add_line(line)

                        mid_x = conversation_center.x
                        mid_y = conversation_center.y

                # Add conversation topic text
                if len(participants) >= 2:
                    topic = conv_data["topic"]
                    # Truncate long topics
                    if len(topic) > 40:
                        topic = topic[:37] + "..."

                    # Add group size indicator for group conversations
                    if len(participants) > 2:
                        topic_text = f"[{len(participants)}] {topic}"
                    else:
                        topic_text = topic

                    self.ax.text(
                        mid_x,
                        mid_y,
                        topic_text,
                        ha="center",
                        va="center",
                        fontsize=7,
                        style="italic",
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="purple",
                            alpha=0.2,
                            edgecolor="purple",
                        ),
                    )

            # Update title with current stats
            active_count = len(active_conversations)
            total_agents = len(self.agents)
            self.ax.set_title(
                f"Agent Map Simulation - {total_agents} Agents, {active_count} Active Conversations",
                fontsize=14,
                fontweight="bold",
            )

            # Refresh the plot
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        except Exception:
            # Silently handle visualization errors to not interrupt simulation
            pass

    def start_live_visualization(self, update_interval: float = 2.0):
        """
        Start live visualization that updates automatically.

        Args:
            update_interval: How often to update the visualization in seconds
        """
        if self.fig is None:
            logger.info("📊 Setting up visualization...")
            self.setup_visualization()

        if self.fig is None:
            logger.error(
                "❌ Could not set up visualization. Running simulation without graphics."
            )
            logger.info("💡 Try installing tkinter: pip install tk")
            return

        try:

            def animate(frame):
                try:
                    self.update_visualization()
                    return []
                except Exception as e:
                    logger.exception(
                        f"⚠️  Error updating visualization: {str(e)}"
                    )
                    return []

            # Create animation
            self.animation = FuncAnimation(
                self.fig,
                animate,
                interval=int(update_interval * 1000),
                blit=False,
                repeat=True,
            )

            logger.success(
                "🎬 Live visualization started - check your screen for the simulation window!"
            )
            logger.info(
                "📊 The visualization will update automatically every few seconds"
            )

            # Try to bring window to front
            try:
                self.fig.canvas.manager.window.wm_attributes(
                    "-topmost", 1
                )
                self.fig.canvas.manager.window.wm_attributes(
                    "-topmost", 0
                )
            except:
                pass  # Not all backends support this

            plt.show(block=False)

        except Exception as e:
            logger.exception(
                f"❌ Error starting live visualization: {str(e)}"
            )
            logger.warning(
                "📊 Simulation will continue without live visualization"
            )

    def save_conversation_summary(self, filename: str = None):
        """
        Save a summary of all conversations to a file.

        Args:
            filename: Output filename (auto-generated if not provided)
        """
        try:
            if filename is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"conversation_summary_{timestamp}.txt"

            with open(filename, "w", encoding="utf-8") as f:
                f.write(
                    "AGENT MAP SIMULATION - CONVERSATION SUMMARY\n"
                )
                f.write("=" * 50 + "\n\n")

                # Write simulation info
                try:
                    state = self.get_simulation_state()
                    f.write(
                        f"Map Size: {state['map_size'][0]}x{state['map_size'][1]}\n"
                    )
                    f.write(f"Total Agents: {len(state['agents'])}\n")
                    f.write(
                        f"Total Conversations: {state['total_conversations']}\n\n"
                    )
                except Exception as e:
                    logger.exception(
                        f"❌ Error writing simulation state: {str(e)}"
                    )
                    f.write("Error retrieving simulation state\n\n")

                # Write agent summaries
                try:
                    f.write("AGENT SUMMARIES:\n")
                    f.write("-" * 20 + "\n")
                    for name, agent_state in self.agents.items():
                        try:
                            f.write(f"\n{name}:\n")
                            f.write(
                                f"  Position: ({agent_state.position.x:.1f}, {agent_state.position.y:.1f})\n"
                            )
                            f.write(
                                f"  Conversations: {len(agent_state.conversation_history)}\n"
                            )

                            if agent_state.conversation_history:
                                f.write("  Recent Conversations:\n")
                                for i, conv in enumerate(
                                    agent_state.conversation_history[
                                        -3:
                                    ],
                                    1,
                                ):  # Last 3 conversations
                                    try:
                                        f.write(
                                            f"    {i}. With {conv['partner']} - {conv['topic'][:50]}...\n"
                                        )
                                    except Exception as e:
                                        logger.exception(
                                            f"❌ Error writing conversation entry: {str(e)}"
                                        )
                                        f.write(
                                            f"    {i}. [Error reading conversation]\n"
                                        )
                        except Exception as e:
                            logger.exception(
                                f"❌ Error writing agent summary for {name}: {str(e)}"
                            )
                            f.write(
                                f"  [Error writing summary for {name}]\n"
                            )
                except Exception as e:
                    logger.exception(
                        f"❌ Error writing agent summaries: {str(e)}"
                    )
                    f.write("Error writing agent summaries\n")

                # Write detailed conversation histories
                try:
                    f.write("\n\nDETAILED CONVERSATION HISTORIES:\n")
                    f.write("-" * 35 + "\n")

                    for (
                        conv_id,
                        conv_data,
                    ) in (
                        self.conversation_manager.active_conversations.items()
                    ):
                        try:
                            if (
                                conv_data["status"] == "completed"
                                and "history" in conv_data
                            ):
                                # Updated to handle both old and new conversation format
                                if "participants" in conv_data:
                                    participant_names = [
                                        p.agent.agent_name
                                        for p in conv_data[
                                            "participants"
                                        ]
                                    ]
                                    f.write(
                                        f"\nConversation: {', '.join(participant_names)}\n"
                                    )
                                else:
                                    # Legacy format
                                    f.write(
                                        f"\nConversation: {conv_data.get('agent1', {}).get('agent', {}).get('agent_name', 'Unknown')} & {conv_data.get('agent2', {}).get('agent', {}).get('agent_name', 'Unknown')}\n"
                                    )

                                f.write(
                                    f"Topic: {conv_data.get('topic', 'Unknown')}\n"
                                )
                                f.write(
                                    f"Duration: {conv_data.get('end_time', 0) - conv_data.get('start_time', 0):.1f} seconds\n"
                                )
                                f.write("History:\n")

                                if isinstance(
                                    conv_data["history"], list
                                ):
                                    for entry in conv_data["history"]:
                                        f.write(f"  {entry}\n")
                                else:
                                    f.write(
                                        f"  {conv_data['history']}\n"
                                    )
                                f.write("\n" + "-" * 40 + "\n")
                        except Exception as e:
                            logger.exception(
                                f"❌ Error writing conversation {conv_id}: {str(e)}"
                            )
                            f.write(
                                f"\n[Error writing conversation {conv_id}]\n"
                            )
                except Exception as e:
                    logger.exception(
                        f"❌ Error writing conversation histories: {str(e)}"
                    )
                    f.write("Error writing conversation histories\n")

            logger.success(
                f"💾 Conversation summary saved to: {filename}"
            )
            return filename
        except Exception as e:
            logger.exception(
                f"❌ Critical error saving conversation summary: {str(e)}"
            )
            return None

    def _generate_simulation_results(
        self, task: str, duration: float
    ) -> Dict[str, Any]:
        """
        Generate a summary of simulation results.

        Args:
            task: The task that was being discussed
            duration: How long the simulation ran

        Returns:
            Dictionary containing simulation results
        """
        self.get_simulation_state()

        # Collect conversation statistics
        total_conversations = len(
            self.conversation_manager.active_conversations
        )
        completed_conversations = sum(
            1
            for conv in self.conversation_manager.active_conversations.values()
            if conv["status"] == "completed"
        )

        # Collect agent participation stats
        agent_stats = {}
        for name, agent_state in self.agents.items():
            agent_stats[name] = {
                "total_conversations": len(
                    agent_state.conversation_history
                ),
                "final_position": (
                    agent_state.position.x,
                    agent_state.position.y,
                ),
                "partners_met": list(
                    set(
                        conv["partner"]
                        for conv in agent_state.conversation_history
                    )
                ),
            }

        # Calculate conversation quality metrics
        avg_conversation_length = 0
        if completed_conversations > 0:
            total_loops = sum(
                conv.get("conversation_loops", 0)
                for conv in self.conversation_manager.active_conversations.values()
                if conv["status"] == "completed"
            )
            avg_conversation_length = (
                total_loops / completed_conversations
            )

        return {
            "task": task,
            "duration_seconds": duration,
            "total_agents": len(self.agents),
            "total_conversations": total_conversations,
            "completed_conversations": completed_conversations,
            "average_conversation_length": avg_conversation_length,
            "agent_statistics": agent_stats,
            "map_dimensions": (self.map_width, self.map_height),
            "simulation_settings": {
                "proximity_threshold": self.proximity_threshold,
                "update_interval": self.update_interval,
            },
        }

    def run(
        self,
        task: str,
        duration: int = 300,
        with_visualization: bool = True,
        update_interval: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Run the simulation with a specific task for agents to discuss.

        Args:
            task: The main topic/task for agents to discuss when they meet
            duration: How long to run the simulation in seconds (default: 5 minutes)
            with_visualization: Whether to show live visualization
            update_interval: How often to update the visualization in seconds

        Returns:
            Dictionary containing simulation results and conversation summaries
        """
        if len(self.agents) == 0:
            logger.error("❌ No agents added to the simulation")
            raise ValueError(
                "No agents added to the simulation. Add agents first using add_agent()"
            )

        logger.debug(
            f"🔍 Validating {len(self.agents)} agents for simulation"
        )

        logger.info("🚀 Starting Agent Map Simulation")
        logger.info(f"📋 Task: {task}")
        logger.info(f"⏱️  Duration: {duration} seconds")
        logger.info(f"👥 Agents: {len(self.agents)}")
        logger.info("=" * 60)

        # Set the task for this simulation run
        self.current_task = task
        self.task_mode = True

        # Set up visualization if requested
        if with_visualization:
            logger.info("📊 Setting up visualization...")
            try:
                self.setup_visualization()
                if self.fig is not None:
                    logger.info("🎬 Starting live visualization...")
                    try:
                        self.start_live_visualization(
                            update_interval=update_interval
                        )
                    except Exception as e:
                        logger.exception(
                            f"⚠️  Visualization error: {str(e)}"
                        )
                        logger.warning(
                            "📊 Continuing without visualization..."
                        )
                else:
                    logger.warning(
                        "📊 Visualization not available, running text-only simulation"
                    )
            except Exception as e:
                logger.exception(
                    f"❌ Failed to setup visualization: {str(e)}"
                )
                logger.warning(
                    "📊 Continuing without visualization..."
                )

        # Start the simulation
        self.start_simulation()

        logger.info(
            f"\n🏃 Simulation running for {duration} seconds..."
        )
        logger.info(
            "💬 Agents will discuss the specified task when they meet"
        )
        logger.info("📊 Status updates every 10 seconds")
        logger.info("⏹️  Press Ctrl+C to stop early")
        logger.info("=" * 60)

        start_time = time.time()
        last_status_time = start_time

        try:
            while (
                time.time() - start_time
            ) < duration and self.running:
                time.sleep(1)

                # Update visualization if available
                if with_visualization and self.fig is not None:
                    try:
                        self.update_visualization()
                    except:
                        pass  # Ignore visualization errors

                # Print status every 10 seconds
                current_time = time.time()
                if current_time - last_status_time >= 10:
                    elapsed = int(current_time - start_time)
                    remaining = max(0, duration - elapsed)
                    logger.info(
                        f"\n⏰ Elapsed: {elapsed}s | Remaining: {remaining}s"
                    )
                    self.print_status()
                    last_status_time = current_time

        except KeyboardInterrupt:
            logger.warning("\n⏹️  Simulation stopped by user")

        except Exception as e:
            logger.exception(
                f"❌ Unexpected error during simulation: {str(e)}"
            )

        finally:
            # Stop the simulation
            self.stop_simulation()

            # Reset task mode
            self.task_mode = False
            self.current_task = None

            logger.success("\n🏁 Simulation completed!")

            # Generate and return results
            results = self._generate_simulation_results(
                task, time.time() - start_time
            )

            # Save detailed summary
            filename = self.save_conversation_summary()
            results["summary_file"] = filename

            logger.info(
                f"📄 Detailed conversation log saved to: {filename}"
            )

            return results
