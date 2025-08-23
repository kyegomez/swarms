"""
Board of Directors Swarm Implementation

This module implements a Board of Directors feature as an alternative to the Director feature
in the Swarms Framework. The Board of Directors operates as a collective decision-making body
that can be enabled manually through configuration.

The implementation follows the Swarms philosophy of:
- Readable code with comprehensive type annotations and documentation
- Performance optimization through concurrency and parallelism
- Simplified abstractions for multi-agent collaboration

Flow:
1. User provides a task
2. Board of Directors convenes to discuss and create a plan
3. Board distributes orders to agents through voting and consensus
4. Agents execute tasks and report back to the board
5. Board evaluates results and issues new orders if needed (up to max_loops)
6. All context and conversation history is preserved throughout the process
"""

import json
import os
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.ma_utils import list_all_agents
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.output_types import OutputType

# Initialize logger for Board of Directors swarm
board_logger = initialize_logger(
    log_folder="board_of_directors_swarm"
)


# ============================================================================
# BOARD OF DIRECTORS CONFIGURATION
# ============================================================================


class BoardConfigModel(BaseModel):
    """
    Configuration model for Board of Directors feature.

    This model defines all configurable parameters for the Board of Directors
    feature, including feature status, board composition, and operational settings.

    Attributes:
        board_feature_enabled: Whether the Board of Directors feature is enabled globally
        default_board_size: Default number of board members when creating a new board
        decision_threshold: Threshold for majority decisions (0.0-1.0)
        enable_voting: Enable voting mechanisms for board decisions
        enable_consensus: Enable consensus-building mechanisms
        default_board_model: Default model for board member agents
        verbose_logging: Enable verbose logging for board operations
        max_board_meeting_duration: Maximum duration for board meetings in seconds
        auto_fallback_to_director: Automatically fall back to Director mode if Board fails
        custom_board_templates: Custom board templates for different use cases
    """

    # Board composition
    default_board_size: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Default number of board members when creating a new board.",
    )

    # Operational settings
    decision_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Threshold for majority decisions (0.0-1.0).",
    )

    enable_voting: bool = Field(
        default=True,
        description="Enable voting mechanisms for board decisions.",
    )

    enable_consensus: bool = Field(
        default=True,
        description="Enable consensus-building mechanisms.",
    )

    # Model settings
    default_board_model: str = Field(
        default="gpt-4o-mini",
        description="Default model for board member agents.",
    )

    # Logging and monitoring
    verbose_logging: bool = Field(
        default=False,
        description="Enable verbose logging for board operations.",
    )

    # Performance settings
    max_board_meeting_duration: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Maximum duration for board meetings in seconds.",
    )

    # Integration settings
    auto_fallback_to_director: bool = Field(
        default=True,
        description="Automatically fall back to Director mode if Board fails.",
    )

    # Custom board templates
    custom_board_templates: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Custom board templates for different use cases.",
    )


@dataclass
class BoardConfig:
    """
    Board of Directors configuration manager.

    This class manages the configuration for the Board of Directors feature,
    including loading from environment variables, configuration files, and
    providing default values.

    Attributes:
        config_file_path: Optional path to configuration file
        config_data: Optional configuration data dictionary
        config: The current configuration model instance
    """

    config_file_path: Optional[str] = None
    config_data: Optional[Dict[str, Any]] = None
    config: BoardConfigModel = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the configuration after object creation."""
        self._load_config()

    def _load_config(self) -> None:
        """
        Load configuration from various sources.

        Priority order:
        1. Environment variables
        2. Configuration file
        3. Default values

        Raises:
            Exception: If configuration loading fails
        """
        try:
            # Start with default configuration
            self.config = BoardConfigModel()

            # Load from configuration file if specified
            if self.config_file_path and os.path.exists(
                self.config_file_path
            ):
                self._load_from_file()

            # Override with explicit config data
            if self.config_data:
                self._load_from_dict(self.config_data)

        except Exception as e:
            logger.error(
                f"Failed to load Board of Directors configuration: {str(e)}"
            )
            raise

    def _load_from_file(self) -> None:
        """
        Load configuration from file.

        Raises:
            Exception: If file loading fails
        """
        try:
            import yaml

            with open(self.config_file_path, "r") as f:
                file_config = yaml.safe_load(f)
                self._load_from_dict(file_config)
                logger.info(
                    f"Loaded Board of Directors config from: {self.config_file_path}"
                )
        except Exception as e:
            logger.warning(
                f"Failed to load config file {self.config_file_path}: {e}"
            )
            raise

    def _load_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Load configuration from dictionary.

        Args:
            config_dict: Dictionary containing configuration values

        Raises:
            ValueError: If configuration values are invalid
        """
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                try:
                    setattr(self.config, key, value)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to set config {key}: {e}")
                    raise ValueError(
                        f"Invalid configuration value for {key}: {e}"
                    )

    def get_config(self) -> BoardConfigModel:
        """
        Get the current configuration.

        Returns:
            BoardConfigModel: The current configuration
        """
        return self.config

    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update the configuration with new values.

        Args:
            updates: Dictionary of configuration updates

        Raises:
            ValueError: If any update values are invalid
        """
        try:
            self._load_from_dict(updates)
        except ValueError as e:
            logger.error(f"Failed to update configuration: {e}")
            raise

    def save_config(self, file_path: Optional[str] = None) -> None:
        """
        Save the current configuration to a file.

        Args:
            file_path: Optional file path to save to (uses config_file_path if not provided)

        Raises:
            Exception: If saving fails
        """
        save_path = file_path or self.config_file_path
        if not save_path:
            logger.warning(
                "No file path specified for saving configuration"
            )
            return

        try:
            import yaml

            # Convert config to dictionary
            config_dict = self.config.model_dump()

            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, "w") as f:
                yaml.dump(
                    config_dict, f, default_flow_style=False, indent=2
                )

            logger.info(
                f"Saved Board of Directors config to: {save_path}"
            )
        except Exception as e:
            logger.error(f"Failed to save config to {save_path}: {e}")
            raise

    @lru_cache(maxsize=128)
    def get_default_board_template(
        self, template_name: str = "standard"
    ) -> Dict[str, Any]:
        """
        Get a default board template.

        This method provides predefined board templates for common use cases.
        Templates are cached for improved performance.

        Args:
            template_name: Name of the template to retrieve

        Returns:
            Dict[str, Any]: Board template configuration
        """
        templates = {
            "standard": {
                "roles": [
                    {
                        "name": "Chairman",
                        "weight": 1.5,
                        "expertise": ["leadership", "strategy"],
                    },
                    {
                        "name": "Vice-Chairman",
                        "weight": 1.2,
                        "expertise": ["operations", "coordination"],
                    },
                    {
                        "name": "Secretary",
                        "weight": 1.0,
                        "expertise": [
                            "documentation",
                            "communication",
                        ],
                    },
                ]
            },
            "executive": {
                "roles": [
                    {
                        "name": "CEO",
                        "weight": 2.0,
                        "expertise": [
                            "executive_leadership",
                            "strategy",
                        ],
                    },
                    {
                        "name": "CFO",
                        "weight": 1.5,
                        "expertise": ["finance", "risk_management"],
                    },
                    {
                        "name": "CTO",
                        "weight": 1.5,
                        "expertise": ["technology", "innovation"],
                    },
                    {
                        "name": "COO",
                        "weight": 1.3,
                        "expertise": ["operations", "efficiency"],
                    },
                ]
            },
            "advisory": {
                "roles": [
                    {
                        "name": "Lead_Advisor",
                        "weight": 1.3,
                        "expertise": ["strategy", "consulting"],
                    },
                    {
                        "name": "Technical_Advisor",
                        "weight": 1.2,
                        "expertise": ["technology", "architecture"],
                    },
                    {
                        "name": "Business_Advisor",
                        "weight": 1.2,
                        "expertise": ["business", "market_analysis"],
                    },
                    {
                        "name": "Legal_Advisor",
                        "weight": 1.1,
                        "expertise": ["legal", "compliance"],
                    },
                ]
            },
            "minimal": {
                "roles": [
                    {
                        "name": "Chairman",
                        "weight": 1.0,
                        "expertise": ["leadership"],
                    },
                    {
                        "name": "Member",
                        "weight": 1.0,
                        "expertise": ["general"],
                    },
                ]
            },
        }

        # Check custom templates first
        if template_name in self.config.custom_board_templates:
            return self.config.custom_board_templates[template_name]

        # Return standard template if requested template not found
        return templates.get(template_name, templates["standard"])

    def validate_config(self) -> List[str]:
        """
        Validate the current configuration.

        This method performs comprehensive validation of the configuration
        to ensure all values are within acceptable ranges and constraints.

        Returns:
            List[str]: List of validation errors (empty if valid)
        """
        errors = []

        try:
            # Validate the configuration model
            self.config.model_validate(self.config.model_dump())
        except Exception as e:
            errors.append(f"Configuration validation failed: {e}")

        # Additional custom validations
        if self.config.decision_threshold < 0.5:
            errors.append(
                "Decision threshold should be at least 0.5 for meaningful majority decisions"
            )

        if self.config.default_board_size < 2:
            errors.append(
                "Board size should be at least 2 for meaningful discussions"
            )

        if self.config.max_board_meeting_duration < 60:
            errors.append(
                "Board meeting duration should be at least 60 seconds"
            )

        return errors


# Global configuration instance
_board_config: Optional[BoardConfig] = None


@lru_cache(maxsize=1)
def get_board_config(
    config_file_path: Optional[str] = None,
) -> BoardConfig:
    """
    Get the global Board of Directors configuration instance.

    This function provides a singleton pattern for accessing the Board of Directors
    configuration. The configuration is cached for improved performance.

    Args:
        config_file_path: Optional path to configuration file

    Returns:
        BoardConfig: The global configuration instance
    """
    global _board_config

    if _board_config is None:
        _board_config = BoardConfig(config_file_path=config_file_path)

    return _board_config


def create_default_config_file(
    file_path: str = "swarms_board_config.yaml",
) -> None:
    """
    Create a default configuration file.

    This function creates a default Board of Directors configuration file
    with recommended settings.

    Args:
        file_path: Path where to create the configuration file
    """
    default_config = {
        "board_feature_enabled": False,
        "default_board_size": 3,
        "decision_threshold": 0.6,
        "enable_voting": True,
        "enable_consensus": True,
        "default_board_model": "gpt-4o-mini",
        "verbose_logging": False,
        "max_board_meeting_duration": 300,
        "auto_fallback_to_director": True,
        "custom_board_templates": {},
    }

    config = BoardConfig(
        config_file_path=file_path, config_data=default_config
    )
    config.save_config(file_path)

    logger.info(
        f"Created default Board of Directors config file: {file_path}"
    )


def set_board_size(
    size: int, config_file_path: Optional[str] = None
) -> None:
    """
    Set the default board size.

    Args:
        size: The default board size (1-10)
        config_file_path: Optional path to save the configuration
    """
    if not 1 <= size <= 10:
        raise ValueError("Board size must be between 1 and 10")

    config = get_board_config(config_file_path)
    config.update_config({"default_board_size": size})

    if config_file_path:
        config.save_config(config_file_path)

    logger.info(f"Default board size set to: {size}")


def set_decision_threshold(
    threshold: float, config_file_path: Optional[str] = None
) -> None:
    """
    Set the decision threshold for majority decisions.

    Args:
        threshold: The decision threshold (0.0-1.0)
        config_file_path: Optional path to save the configuration
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Decision threshold must be between 0.0 and 1.0"
        )

    config = get_board_config(config_file_path)
    config.update_config({"decision_threshold": threshold})

    if config_file_path:
        config.save_config(config_file_path)

    logger.info(f"Decision threshold set to: {threshold}")


def set_board_model(
    model: str, config_file_path: Optional[str] = None
) -> None:
    """
    Set the default board model.

    Args:
        model: The default model name for board members
        config_file_path: Optional path to save the configuration
    """
    config = get_board_config(config_file_path)
    config.update_config({"default_board_model": model})

    if config_file_path:
        config.save_config(config_file_path)

    logger.info(f"Default board model set to: {model}")


def enable_verbose_logging(
    config_file_path: Optional[str] = None,
) -> None:
    """
    Enable verbose logging for board operations.

    Args:
        config_file_path: Optional path to save the configuration
    """
    config = get_board_config(config_file_path)
    config.update_config({"verbose_logging": True})

    if config_file_path:
        config.save_config(config_file_path)

    logger.info("Verbose logging enabled for Board of Directors")


def disable_verbose_logging(
    config_file_path: Optional[str] = None,
) -> None:
    """
    Disable verbose logging for board operations.

    Args:
        config_file_path: Optional path to save the configuration
    """
    config = get_board_config(config_file_path)
    config.update_config({"verbose_logging": False})

    if config_file_path:
        config.save_config(config_file_path)

    logger.info("Verbose logging disabled for Board of Directors")


# ============================================================================
# BOARD OF DIRECTORS IMPLEMENTATION
# ============================================================================


class BoardMemberRole(str, Enum):
    """Enumeration of possible board member roles.

    This enum defines the various roles that board members can have within
    the Board of Directors swarm. Each role has specific responsibilities
    and voting weights associated with it.

    Attributes:
        CHAIRMAN: Primary leader responsible for board meetings and final decisions
        VICE_CHAIRMAN: Secondary leader who supports the chairman
        SECRETARY: Responsible for documentation and meeting minutes
        TREASURER: Manages financial aspects and resource allocation
        MEMBER: General board member with specific expertise
        EXECUTIVE_DIRECTOR: Executive-level board member with operational authority
    """

    CHAIRMAN = "chairman"
    VICE_CHAIRMAN = "vice_chairman"
    SECRETARY = "secretary"
    TREASURER = "treasurer"
    MEMBER = "member"
    EXECUTIVE_DIRECTOR = "executive_director"


class BoardDecisionType(str, Enum):
    """Enumeration of board decision types.

    This enum defines the different types of decisions that can be made
    by the Board of Directors, including voting mechanisms and consensus
    approaches.

    Attributes:
        UNANIMOUS: All board members agree on the decision
        MAJORITY: More than 50% of votes are in favor
        CONSENSUS: General agreement without formal voting
        CHAIRMAN_DECISION: Final decision made by the chairman
    """

    UNANIMOUS = "unanimous"
    MAJORITY = "majority"
    CONSENSUS = "consensus"
    CHAIRMAN_DECISION = "chairman_decision"


@dataclass
class BoardMember:
    """
    Represents a member of the Board of Directors.

    This dataclass encapsulates all information about a board member,
    including their agent representation, role, voting weight, and
    areas of expertise.

    Attributes:
        agent: The agent representing this board member
        role: The role of this board member within the board
        voting_weight: The weight of this member's vote (default: 1.0)
        expertise_areas: Areas of expertise for this board member
    """

    agent: Agent
    role: BoardMemberRole
    voting_weight: float = 1.0
    expertise_areas: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize default values after object creation.

        This method ensures that the expertise_areas list is properly
        initialized as an empty list if not provided.
        """
        if self.expertise_areas is None:
            self.expertise_areas = []


class BoardOrder(BaseModel):
    """
    Represents an order issued by the Board of Directors.

    This model defines the structure of orders that the board issues
    to worker agents, including task assignments, priorities, and
    deadlines.

    Attributes:
        agent_name: The name of the agent to which the task is assigned
        task: The specific task to be executed by the assigned agent
        priority: Priority level of the task (1-5, where 1 is highest)
        deadline: Optional deadline for task completion
        assigned_by: The board member who assigned this task
    """

    agent_name: str = Field(
        ...,
        description="Specifies the name of the agent to which the task is assigned.",
    )
    task: str = Field(
        ...,
        description="Defines the specific task to be executed by the assigned agent.",
    )
    priority: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Priority level of the task (1-5, where 1 is highest priority).",
    )
    deadline: Optional[str] = Field(
        default=None,
        description="Optional deadline for task completion.",
    )
    assigned_by: str = Field(
        default="Board of Directors",
        description="The board member who assigned this task.",
    )


class BoardDecision(BaseModel):
    """
    Represents a decision made by the Board of Directors.

    This model tracks the details of decisions made by the board,
    including voting results, decision types, and reasoning.

    Attributes:
        decision_type: The type of decision (unanimous, majority, etc.)
        decision: The actual decision made
        votes_for: Number of votes in favor
        votes_against: Number of votes against
        abstentions: Number of abstentions
        reasoning: The reasoning behind the decision
    """

    decision_type: BoardDecisionType = Field(
        ...,
        description="The type of decision made by the board.",
    )
    decision: str = Field(
        ...,
        description="The actual decision made by the board.",
    )
    votes_for: int = Field(
        default=0,
        ge=0,
        description="Number of votes in favor of the decision.",
    )
    votes_against: int = Field(
        default=0,
        ge=0,
        description="Number of votes against the decision.",
    )
    abstentions: int = Field(
        default=0,
        ge=0,
        description="Number of abstentions.",
    )
    reasoning: str = Field(
        default="",
        description="The reasoning behind the decision.",
    )


class BoardSpec(BaseModel):
    """
    Specification for Board of Directors operations.

    This model represents the complete output of a board meeting,
    including the plan, orders, decisions, and meeting summary.

    Attributes:
        plan: The overall plan created by the board
        orders: List of orders issued by the board
        decisions: List of decisions made by the board
        meeting_summary: Summary of the board meeting
    """

    plan: str = Field(
        ...,
        description="Outlines the sequence of actions to be taken by the swarm as decided by the board.",
    )
    orders: List[BoardOrder] = Field(
        ...,
        description="A collection of task assignments to specific agents within the swarm.",
    )
    decisions: List[BoardDecision] = Field(
        default_factory=list,
        description="List of decisions made by the board during the meeting.",
    )
    meeting_summary: str = Field(
        default="",
        description="Summary of the board meeting and key outcomes.",
    )


class BoardOfDirectorsSwarm:
    """
    A hierarchical swarm of agents with a Board of Directors that orchestrates tasks.

    The Board of Directors operates as a collective decision-making body that can be
    enabled manually through configuration. It provides an alternative to the single
    Director approach with more democratic and collaborative decision-making.

    The workflow follows a hierarchical pattern:
    1. Task is received and sent to the Board of Directors
    2. Board convenes to discuss and create a plan through voting and consensus
    3. Board distributes orders to agents based on collective decisions
    4. Agents execute tasks and report back to the board
    5. Board evaluates results and issues new orders if needed (up to max_loops)
    6. All context and conversation history is preserved throughout the process

    Attributes:
        name: The name of the swarm
        description: A description of the swarm
        board_members: List of board members with their roles and expertise
        agents: A list of agents within the swarm
        max_loops: The maximum number of feedback loops between the board and agents
        output_type: The format in which to return the output (dict, str, or list)
        board_model_name: The model name for board member agents
        verbose: Enable detailed logging with loguru
        add_collaboration_prompt: Add collaboration prompts to agents
        board_feedback_on: Enable board feedback on agent outputs
        decision_threshold: Threshold for majority decisions (0.0-1.0)
        enable_voting: Enable voting mechanisms for board decisions
        enable_consensus: Enable consensus-building mechanisms
        max_workers: Maximum number of workers for parallel execution
    """

    def __init__(
        self,
        name: str = "BoardOfDirectorsSwarm",
        description: str = "Distributed task swarm with collective decision-making",
        board_members: Optional[List[BoardMember]] = None,
        agents: Optional[List[Union[Agent, Callable, Any]]] = None,
        max_loops: int = 1,
        output_type: OutputType = "dict-all-except-first",
        board_model_name: str = "gpt-4o-mini",
        verbose: bool = False,
        add_collaboration_prompt: bool = True,
        board_feedback_on: bool = True,
        decision_threshold: float = 0.6,
        enable_voting: bool = True,
        enable_consensus: bool = True,
        max_workers: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Board of Directors Swarm with the given parameters.

        Args:
            name: The name of the swarm
            description: A description of the swarm
            board_members: List of board members with their roles and expertise
            agents: A list of agents within the swarm
            max_loops: The maximum number of feedback loops between the board and agents
            output_type: The format in which to return the output (dict, str, or list)
            board_model_name: The model name for board member agents
            verbose: Enable detailed logging with loguru
            add_collaboration_prompt: Add collaboration prompts to agents
            board_feedback_on: Enable board feedback on agent outputs
            decision_threshold: Threshold for majority decisions (0.0-1.0)
            enable_voting: Enable voting mechanisms for board decisions
            enable_consensus: Enable consensus-building mechanisms
            max_workers: Maximum number of workers for parallel execution
            *args: Additional positional arguments passed to BaseSwarm
            **kwargs: Additional keyword arguments passed to BaseSwarm

        Raises:
            ValueError: If critical requirements are not met during initialization
        """
        self.name = name
        self.description = description
        self.board_members = board_members or []
        self.agents = agents or []
        self.max_loops = max_loops
        self.output_type = output_type
        self.board_model_name = board_model_name
        self.verbose = verbose
        self.add_collaboration_prompt = add_collaboration_prompt
        self.board_feedback_on = board_feedback_on
        self.decision_threshold = decision_threshold
        self.enable_voting = enable_voting
        self.enable_consensus = enable_consensus
        self.max_workers = max_workers
        self.max_workers = os.cpu_count()

        # Initialize the swarm
        self._init_board_swarm()

    def _init_board_swarm(self) -> None:
        """
        Initialize the Board of Directors swarm.

        This method sets up the board members, initializes the conversation,
        performs reliability checks, and prepares the board for operation.

        Raises:
            ValueError: If reliability checks fail
        """
        if self.verbose:
            board_logger.info(
                f"🚀 Initializing Board of Directors Swarm: {self.name}"
            )
            board_logger.info(
                f"📊 Configuration - Max loops: {self.max_loops}"
            )

        self.conversation = Conversation(time_enabled=False)

        # Perform reliability checks
        self._perform_reliability_checks()

        # Setup board members if not provided
        if not self.board_members:
            self._setup_default_board()

        # Add context to board members
        self._add_context_to_board()

        if self.verbose:
            board_logger.success(
                f"✅ Board of Directors Swarm initialized successfully: {self.name}"
            )

    def _setup_default_board(self) -> None:
        """
        Set up a default Board of Directors if none is provided.

        Creates a basic board structure with Chairman, Vice Chairman, and Secretary roles.
        This method is called automatically if no board members are provided during initialization.
        """
        if self.verbose:
            board_logger.info(
                "🎯 Setting up default Board of Directors"
            )

        # Create default board members
        chairman = Agent(
            agent_name="Chairman",
            agent_description="Chairman of the Board responsible for leading meetings and making final decisions",
            model_name=self.board_model_name,
            max_loops=1,
            system_prompt=self._get_chairman_prompt(),
        )

        vice_chairman = Agent(
            agent_name="Vice-Chairman",
            agent_description="Vice Chairman who supports the Chairman and leads in their absence",
            model_name=self.board_model_name,
            max_loops=1,
            system_prompt=self._get_vice_chairman_prompt(),
        )

        secretary = Agent(
            agent_name="Secretary",
            agent_description="Board Secretary responsible for documentation and meeting minutes",
            model_name=self.board_model_name,
            max_loops=1,
            system_prompt=self._get_secretary_prompt(),
        )

        self.board_members = [
            BoardMember(
                chairman,
                BoardMemberRole.CHAIRMAN,
                1.5,
                ["leadership", "strategy"],
            ),
            BoardMember(
                vice_chairman,
                BoardMemberRole.VICE_CHAIRMAN,
                1.2,
                ["operations", "coordination"],
            ),
            BoardMember(
                secretary,
                BoardMemberRole.SECRETARY,
                1.0,
                ["documentation", "communication"],
            ),
        ]

        if self.verbose:
            board_logger.success(
                "✅ Default Board of Directors setup completed"
            )

    def _get_chairman_prompt(self) -> str:
        """
        Get the system prompt for the Chairman role.

        Returns:
            str: The system prompt defining the Chairman's responsibilities and behavior
        """
        return """You are the Chairman of the Board of Directors. Your responsibilities include:
1. Leading board meetings and discussions
2. Facilitating consensus among board members
3. Making final decisions when consensus cannot be reached
4. Ensuring all board members have an opportunity to contribute
5. Maintaining focus on the organization's goals and objectives
6. Providing strategic direction and oversight

You should be diplomatic, fair, and decisive in your leadership."""

    def _get_vice_chairman_prompt(self) -> str:
        """
        Get the system prompt for the Vice Chairman role.

        Returns:
            str: The system prompt defining the Vice Chairman's responsibilities and behavior
        """
        return """You are the Vice Chairman of the Board of Directors. Your responsibilities include:
1. Supporting the Chairman in leading board meetings
2. Taking leadership when the Chairman is unavailable
3. Coordinating with other board members
4. Ensuring operational efficiency
5. Providing strategic input and analysis
6. Maintaining board cohesion and effectiveness

You should be collaborative, analytical, and supportive in your role."""

    def _get_secretary_prompt(self) -> str:
        """
        Get the system prompt for the Secretary role.

        Returns:
            str: The system prompt defining the Secretary's responsibilities and behavior
        """
        return """You are the Secretary of the Board of Directors. Your responsibilities include:
1. Documenting all board meetings and decisions
2. Maintaining accurate records of board proceedings
3. Ensuring proper communication between board members
4. Tracking action items and follow-ups
5. Providing administrative support to the board
6. Ensuring compliance with governance requirements

You should be thorough, organized, and detail-oriented in your documentation."""

    def _add_context_to_board(self) -> None:
        """
        Add agent context to all board members' conversations.

        This ensures that board members are aware of all available agents
        and their capabilities when making decisions.

        Raises:
            Exception: If context addition fails
        """
        try:
            if self.verbose:
                board_logger.info(
                    "📝 Adding agent context to board members"
                )

            # Add context to each board member
            for board_member in self.board_members:
                list_all_agents(
                    agents=self.agents,
                    conversation=self.conversation,
                    add_to_conversation=True,
                    add_collaboration_prompt=self.add_collaboration_prompt,
                )

            if self.verbose:
                board_logger.success(
                    "✅ Agent context added to board members successfully"
                )

        except Exception as e:
            error_msg = (
                f"❌ Failed to add context to board members: {str(e)}"
            )
            board_logger.error(
                f"{error_msg}\n🔍 Traceback: {traceback.format_exc()}"
            )
            raise

    def _perform_reliability_checks(self) -> None:
        """
        Perform reliability checks for the Board of Directors swarm.

        This method validates critical requirements and configuration
        parameters to ensure the swarm can operate correctly.

        Raises:
            ValueError: If critical requirements are not met
        """
        try:
            if self.verbose:
                board_logger.info(
                    f"🔍 Running reliability checks for swarm: {self.name}"
                )

            if not self.agents or len(self.agents) == 0:
                raise ValueError(
                    "No agents found in the swarm. At least one agent must be provided to create a Board of Directors swarm."
                )

            if self.max_loops <= 0:
                raise ValueError(
                    "Max loops must be greater than 0. Please set a valid number of loops."
                )

            if (
                self.decision_threshold < 0.0
                or self.decision_threshold > 1.0
            ):
                raise ValueError(
                    "Decision threshold must be between 0.0 and 1.0."
                )

            if self.verbose:
                board_logger.success(
                    f"✅ Reliability checks passed for swarm: {self.name}"
                )
                board_logger.info(
                    f"📊 Swarm stats - Agents: {len(self.agents)}, Max loops: {self.max_loops}"
                )

        except Exception as e:
            error_msg = f"❌ Failed reliability checks: {str(e)}\n🔍 Traceback: {traceback.format_exc()}"
            board_logger.error(error_msg)
            raise

    def run_board_meeting(
        self,
        task: str,
        img: Optional[str] = None,
    ) -> BoardSpec:
        """
        Run a board meeting to discuss and decide on the given task.

        This method orchestrates a complete board meeting, including discussion,
        decision-making, and task distribution to worker agents.

        Args:
            task: The task to be discussed and planned by the board
            img: Optional image to be used with the task

        Returns:
            BoardSpec: The board's plan and orders

        Raises:
            Exception: If board meeting execution fails
        """
        try:
            if self.verbose:
                board_logger.info(
                    f"🏛️ Running board meeting with task: {task[:100]}..."
                )

            # Create board meeting prompt
            meeting_prompt = self._create_board_meeting_prompt(task)

            # Run board discussion
            board_discussion = self._conduct_board_discussion(
                meeting_prompt, img
            )

            # Parse board decisions
            board_spec = self._parse_board_decisions(board_discussion)

            # Add to conversation history
            self.conversation.add(
                role="Board of Directors", content=board_discussion
            )

            if self.verbose:
                board_logger.success("✅ Board meeting completed")
                board_logger.debug(
                    f"📋 Board output type: {type(board_spec)}"
                )

            return board_spec

        except Exception as e:
            error_msg = f"❌ Failed to run board meeting: {str(e)}\n🔍 Traceback: {traceback.format_exc()}"
            board_logger.error(error_msg)
            raise

    def _create_board_meeting_prompt(self, task: str) -> str:
        """
        Create a prompt for the board meeting.

        This method generates a comprehensive prompt that guides the board
        through the meeting process, including task discussion, decision-making,
        and task distribution.

        Args:
            task: The task to be discussed

        Returns:
            str: The board meeting prompt
        """
        return f"""BOARD OF DIRECTORS MEETING

TASK: {task}

CONVERSATION HISTORY: {self.conversation.get_str()}

AVAILABLE AGENTS: {[agent.agent_name for agent in self.agents]}

BOARD MEMBERS:
{self._format_board_members_info()}

INSTRUCTIONS:
1. Discuss the task thoroughly as a board
2. Consider all perspectives and expertise areas
3. Reach consensus or majority decision on the approach
4. Create a detailed plan for task execution
5. Assign specific tasks to appropriate agents
6. Document all decisions and reasoning

Please provide your response in the following format:
{{
    "plan": "Detailed plan for task execution",
    "orders": [
        {{
            "agent_name": "Agent Name",
            "task": "Specific task description",
            "priority": 1-5,
            "deadline": "Optional deadline",
            "assigned_by": "Board Member Name"
        }}
    ],
    "decisions": [
        {{
            "decision_type": "unanimous/majority/consensus/chairman_decision",
            "decision": "Description of the decision",
            "votes_for": 0,
            "votes_against": 0,
            "abstentions": 0,
            "reasoning": "Reasoning behind the decision"
        }}
    ],
    "meeting_summary": "Summary of the board meeting and key outcomes"
}}"""

    def _format_board_members_info(self) -> str:
        """
        Format board members information for the prompt.

        This method creates a formatted string containing information about
        all board members, their roles, and expertise areas.

        Returns:
            str: Formatted board members information
        """
        info = []
        for member in self.board_members:
            info.append(
                f"- {member.agent.agent_name} ({member.role.value}): {member.agent.agent_description}"
            )
            if member.expertise_areas:
                info.append(
                    f"  Expertise: {', '.join(member.expertise_areas)}"
                )
        return "\n".join(info)

    def _conduct_board_discussion(
        self, prompt: str, img: Optional[str] = None
    ) -> str:
        """
        Conduct the board discussion using the chairman as the primary speaker.

        This method uses the chairman agent to lead the board discussion
        and generate the meeting output.

        Args:
            prompt: The board meeting prompt
            img: Optional image input

        Returns:
            str: The board discussion output

        Raises:
            ValueError: If no chairman is found in board members
        """
        # Use the chairman to lead the discussion
        chairman = next(
            (
                member.agent
                for member in self.board_members
                if member.role == BoardMemberRole.CHAIRMAN
            ),
            (
                self.board_members[0].agent
                if self.board_members
                else None
            ),
        )

        if not chairman:
            raise ValueError("No chairman found in board members")

        return chairman.run(task=prompt, img=img)

    def _parse_board_decisions(self, board_output: str) -> BoardSpec:
        """
        Parse the board output into a BoardSpec object.

        This method attempts to parse the board discussion output as JSON
        and convert it into a structured BoardSpec object. If parsing fails,
        it returns a basic BoardSpec with the raw output.

        Args:
            board_output: The output from the board discussion

        Returns:
            BoardSpec: Parsed board specification
        """
        try:
            # Try to parse as JSON first
            if isinstance(board_output, str):
                # Try to extract JSON from the response
                json_match = re.search(
                    r"\{.*\}", board_output, re.DOTALL
                )
                if json_match:
                    board_output = json_match.group()

                parsed = json.loads(board_output)
            else:
                parsed = board_output

            # Extract components
            plan = parsed.get("plan", "")
            orders_data = parsed.get("orders", [])
            decisions_data = parsed.get("decisions", [])
            meeting_summary = parsed.get("meeting_summary", "")

            # Create BoardOrder objects
            orders = []
            for order_data in orders_data:
                order = BoardOrder(
                    agent_name=order_data.get("agent_name", ""),
                    task=order_data.get("task", ""),
                    priority=order_data.get("priority", 3),
                    deadline=order_data.get("deadline"),
                    assigned_by=order_data.get(
                        "assigned_by", "Board of Directors"
                    ),
                )
                orders.append(order)

            # Create BoardDecision objects
            decisions = []
            for decision_data in decisions_data:
                decision = BoardDecision(
                    decision_type=BoardDecisionType(
                        decision_data.get(
                            "decision_type", "consensus"
                        )
                    ),
                    decision=decision_data.get("decision", ""),
                    votes_for=decision_data.get("votes_for", 0),
                    votes_against=decision_data.get(
                        "votes_against", 0
                    ),
                    abstentions=decision_data.get("abstentions", 0),
                    reasoning=decision_data.get("reasoning", ""),
                )
                decisions.append(decision)

            return BoardSpec(
                plan=plan,
                orders=orders,
                decisions=decisions,
                meeting_summary=meeting_summary,
            )

        except Exception as e:
            board_logger.error(
                f"Failed to parse board decisions: {str(e)}"
            )
            # Return a basic BoardSpec if parsing fails
            return BoardSpec(
                plan=board_output,
                orders=[],
                decisions=[],
                meeting_summary="Parsing failed, using raw output",
            )

    def step(
        self,
        task: str,
        img: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a single step of the Board of Directors swarm.

        This method runs one complete cycle of board meeting and task execution.
        It includes board discussion, task distribution, and optional feedback.

        Args:
            task: The task to be executed
            img: Optional image input
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Any: The result of the step execution

        Raises:
            Exception: If step execution fails
        """
        try:
            if self.verbose:
                board_logger.info(
                    f"👣 Executing single step for task: {task[:100]}..."
                )

            # Run board meeting
            board_spec = self.run_board_meeting(task=task, img=img)

            if self.verbose:
                board_logger.info(
                    f"📋 Board created plan and {len(board_spec.orders)} orders"
                )

            # Execute the orders
            outputs = self._execute_orders(board_spec.orders)

            if self.verbose:
                board_logger.info(
                    f"⚡ Executed {len(outputs)} orders"
                )

            # Provide board feedback if enabled
            if self.board_feedback_on:
                feedback = self._generate_board_feedback(outputs)
            else:
                feedback = outputs

            if self.verbose:
                board_logger.success("✅ Step completed successfully")

            return feedback

        except Exception as e:
            error_msg = f"❌ Failed to execute step: {str(e)}\n🔍 Traceback: {traceback.format_exc()}"
            board_logger.error(error_msg)
            raise

    def run(
        self,
        task: str,
        img: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Run the Board of Directors swarm for the specified number of loops.

        This method executes the complete swarm workflow, including multiple
        iterations if max_loops is greater than 1. Each iteration includes
        board meeting, task execution, and feedback generation.

        Args:
            task: The task to be executed
            img: Optional image input
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Any: The final result of the swarm execution

        Raises:
            Exception: If swarm execution fails
        """
        try:
            if self.verbose:
                board_logger.info(
                    f"🏛️ Starting Board of Directors swarm execution: {self.name}"
                )
                board_logger.info(f"📋 Task: {task[:100]}...")

            current_loop = 0
            while current_loop < self.max_loops:
                if self.verbose:
                    board_logger.info(
                        f"🔄 Executing loop {current_loop + 1}/{self.max_loops}"
                    )

                # Execute step
                self.step(task=task, img=img, *args, **kwargs)

                # Add to conversation
                self.conversation.add(
                    role="System",
                    content=f"Loop {current_loop + 1} completed",
                )

                current_loop += 1

            if self.verbose:
                board_logger.success(
                    f"🎉 Board of Directors swarm run completed: {self.name}"
                )
                board_logger.info(
                    f"📊 Total loops executed: {current_loop}"
                )

            return history_output_formatter(
                conversation=self.conversation, type=self.output_type
            )

        except Exception as e:
            error_msg = f"❌ Failed to run Board of Directors swarm: {str(e)}\n🔍 Traceback: {traceback.format_exc()}"
            board_logger.error(error_msg)
            raise

    def _generate_board_feedback(self, outputs: List[Any]) -> str:
        """
        Provide feedback from the Board of Directors based on agent outputs.

        This method uses the chairman to review and provide feedback on
        the outputs generated by worker agents.

        Args:
            outputs: List of outputs from agents

        Returns:
            str: Board feedback on the outputs

        Raises:
            ValueError: If no chairman is found for feedback
            Exception: If feedback generation fails
        """
        try:
            if self.verbose:
                board_logger.info("📝 Generating board feedback")

            task = f"History: {self.conversation.get_str()} \n\n"

            # Use the chairman for feedback
            chairman = next(
                (
                    member.agent
                    for member in self.board_members
                    if member.role == BoardMemberRole.CHAIRMAN
                ),
                (
                    self.board_members[0].agent
                    if self.board_members
                    else None
                ),
            )

            if not chairman:
                raise ValueError("No chairman found for feedback")

            feedback_prompt = (
                "You are the Chairman of the Board. Review the outputs generated by all the worker agents "
                "in the previous step. Provide specific, actionable feedback for each agent, highlighting "
                "strengths, weaknesses, and concrete suggestions for improvement. "
                "If any outputs are unclear, incomplete, or could be enhanced, explain exactly how. "
                f"Your feedback should help the agents refine their work in the next iteration. "
                f"Worker Agent Responses: {task}"
            )

            output = chairman.run(task=feedback_prompt)
            self.conversation.add(
                role=chairman.agent_name, content=output
            )

            if self.verbose:
                board_logger.success(
                    "✅ Board feedback generated successfully"
                )

            return output

        except Exception as e:
            error_msg = f"❌ Failed to generate board feedback: {str(e)}\n🔍 Traceback: {traceback.format_exc()}"
            board_logger.error(error_msg)
            raise

    def _call_single_agent(
        self, agent_name: str, task: str, *args: Any, **kwargs: Any
    ) -> Any:
        """
        Call a single agent with the given task.

        This method finds and executes a specific agent with the provided task.
        It includes error handling and logging for agent execution.

        Args:
            agent_name: The name of the agent to call
            task: The task to assign to the agent
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Any: The output from the agent

        Raises:
            ValueError: If the specified agent is not found
            Exception: If agent execution fails
        """
        try:
            if self.verbose:
                board_logger.info(f"📞 Calling agent: {agent_name}")

            # Find agent by name
            agent = None
            for a in self.agents:
                if (
                    hasattr(a, "agent_name")
                    and a.agent_name == agent_name
                ):
                    agent = a
                    break

            if agent is None:
                available_agents = [
                    a.agent_name
                    for a in self.agents
                    if hasattr(a, "agent_name")
                ]
                raise ValueError(
                    f"Agent '{agent_name}' not found in swarm. Available agents: {available_agents}"
                )

            output = agent.run(
                task=f"History: {self.conversation.get_str()} \n\n Task: {task}",
                *args,
                **kwargs,
            )
            self.conversation.add(role=agent_name, content=output)

            if self.verbose:
                board_logger.success(
                    f"✅ Agent {agent_name} completed task successfully"
                )

            return output

        except Exception as e:
            error_msg = f"❌ Failed to call agent {agent_name}: {str(e)}\n🔍 Traceback: {traceback.format_exc()}"
            board_logger.error(error_msg)
            raise

    def _execute_orders(
        self, orders: List[BoardOrder]
    ) -> List[Dict[str, Any]]:
        """
        Execute the orders issued by the Board of Directors.

        This method uses ThreadPoolExecutor to execute multiple orders in parallel,
        improving performance for complex task distributions.

        Args:
            orders: List of board orders to execute

        Returns:
            List[Dict[str, Any]]: List of outputs from executed orders

        Raises:
            Exception: If order execution fails
        """
        try:
            if self.verbose:
                board_logger.info(
                    f"⚡ Executing {len(orders)} board orders"
                )

            # Use ThreadPoolExecutor for parallel execution
            outputs = []
            with ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                # Submit all orders for execution
                future_to_order = {
                    executor.submit(
                        self._execute_single_order, order
                    ): order
                    for order in orders
                }

                # Collect results as they complete
                for future in as_completed(future_to_order):
                    order = future_to_order[future]
                    try:
                        output = future.result()
                        outputs.append(
                            {
                                "agent_name": order.agent_name,
                                "task": order.task,
                                "output": output,
                                "priority": order.priority,
                                "assigned_by": order.assigned_by,
                            }
                        )
                    except Exception as e:
                        board_logger.error(
                            f"Failed to execute order for {order.agent_name}: {str(e)}"
                        )
                        outputs.append(
                            {
                                "agent_name": order.agent_name,
                                "task": order.task,
                                "output": f"Error: {str(e)}",
                                "priority": order.priority,
                                "assigned_by": order.assigned_by,
                            }
                        )

            if self.verbose:
                board_logger.success(
                    f"✅ Executed {len(outputs)} orders successfully"
                )

            return outputs

        except Exception as e:
            error_msg = f"❌ Failed to execute orders: {str(e)}\n🔍 Traceback: {traceback.format_exc()}"
            board_logger.error(error_msg)
            raise

    def _execute_single_order(self, order: BoardOrder) -> Any:
        """
        Execute a single board order.

        This method is a wrapper around _call_single_agent for executing
        individual board orders.

        Args:
            order: The board order to execute

        Returns:
            Any: The output from the executed order
        """
        return self._call_single_agent(
            agent_name=order.agent_name,
            task=order.task,
        )

    def add_board_member(self, board_member: BoardMember) -> None:
        """
        Add a new member to the Board of Directors.

        This method allows dynamic addition of board members after swarm initialization.

        Args:
            board_member: The board member to add
        """
        self.board_members.append(board_member)
        if self.verbose:
            board_logger.info(
                f"✅ Added board member: {board_member.agent.agent_name}"
            )

    def remove_board_member(self, agent_name: str) -> None:
        """
        Remove a board member by agent name.

        This method allows dynamic removal of board members after swarm initialization.

        Args:
            agent_name: The name of the agent to remove from the board
        """
        self.board_members = [
            member
            for member in self.board_members
            if member.agent.agent_name != agent_name
        ]
        if self.verbose:
            board_logger.info(
                f"✅ Removed board member: {agent_name}"
            )

    def get_board_member(
        self, agent_name: str
    ) -> Optional[BoardMember]:
        """
        Get a board member by agent name.

        This method retrieves a specific board member by their agent name.

        Args:
            agent_name: The name of the agent

        Returns:
            Optional[BoardMember]: The board member if found, None otherwise
        """
        for member in self.board_members:
            if member.agent.agent_name == agent_name:
                return member
        return None

    def get_board_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the Board of Directors.

        This method provides a comprehensive summary of the board structure,
        including member information, configuration, and statistics.

        Returns:
            Dict[str, Any]: Summary of the board structure and members
        """
        return {
            "board_name": self.name,
            "total_members": len(self.board_members),
            "members": [
                {
                    "name": member.agent.agent_name,
                    "role": member.role.value,
                    "voting_weight": member.voting_weight,
                    "expertise_areas": member.expertise_areas,
                }
                for member in self.board_members
            ],
            "total_agents": len(self.agents),
            "max_loops": self.max_loops,
            "decision_threshold": self.decision_threshold,
        }
